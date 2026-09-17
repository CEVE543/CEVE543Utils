# Water level records from a tide gauge, and the annual maxima taken from them.
#
# The raw record is what a gauge reports: a level at a time. The annual maximum
# record is one value per year, taken from the raw record and then detrended.
# Both carry the station they came from, so a fit can say what it was fit to.
#
# A record's unit is a type parameter rather than a property of each reading, so
# every level in one record shares it by construction and the levels are stored
# as a plain `Vector{Float64}`. A record of a million readings therefore costs
# the same memory as the bare numbers, and mixing feet into a record of metres
# is a method error rather than a silent conversion.

using DataFrames: DataFrames, DataFrame
using Dates: DateTime, year
using Unitful: Length, Quantity, Units, dimension, uconvert, unit, ustrip, @u_str

"""
    Station(id; name="", datum="MSL")

Where a water level record came from.

`id` is the NOAA CO-OPS station number, `name` is how NOAA labels it, and
`datum` names the vertical reference the levels are measured against (`"MSL"`
for mean sea level, `"NAVD88"`, and so on). A level means nothing without its
datum, so the datum travels with the data rather than living in a comment.

Station ids are listed at <https://tidesandcurrents.noaa.gov/stations.html>.
"""
struct Station
    id::String
    name::String
    datum::String
end

Station(id::AbstractString; name::AbstractString="", datum::AbstractString="MSL") =
    Station(String(id), String(name), String(datum))

function Base.show(io::IO, station::Station)
    label = isempty(station.name) ? station.id : "$(station.name) ($(station.id))"
    return print(io, "$label, $(station.datum) datum")
end

"""
    WaterLevelRecord(station, times, levels, unit)

Every water level reading from one station, in time order.

`levels` holds bare numbers and `unit` says what they are measured in, so the
whole record shares one unit. Indexing gives a reading back with its unit
attached:

    record = load_water_level("8638610")
    length(record)                    # how many readings
    record[1]                         # (time = ..., level = -0.468 m)
    waterlevels(record)               # every level, with units
    ustrip.(waterlevels(record))      # the bare numbers, no copy of the units

[`AnnMaxRecord`](@ref) is built from this.
"""
struct WaterLevelRecord{U<:Units}
    station::Station
    times::Vector{DateTime}
    levels::Vector{Float64}
    unit::U

    function WaterLevelRecord(
        station::Station, times::Vector{DateTime}, levels::Vector{Float64}, unit::U
    ) where {U<:Units}
        length(times) == length(levels) || throw(
            DimensionMismatch(
                "got $(length(times)) times and $(length(levels)) levels"
            ),
        )
        dimension(unit) == dimension(u"m") ||
            throw(ArgumentError("a water level needs a length unit, got $unit"))
        return new{U}(station, times, levels, unit)
    end
end

"""
    AnnMaxRecord(station, years, levels, unit, detrend, baseline)

One water level maximum per year, with the sea-level trend removed.

`levels` holds the detrended maxima as bare numbers in `unit`. `detrend` names
the method that produced them and `baseline` is what was subtracted from each
year, kept so a plot can show the trend that was taken out.

    annmax = AnnMaxRecord(load_water_level("8638610"); detrend=:msl)
    waterlevels(annmax)               # the detrended maxima, with units
    ustrip.(waterlevels(annmax))      # bare numbers for a fit
"""
struct AnnMaxRecord{U<:Units}
    station::Station
    years::Vector{Int}
    levels::Vector{Float64}
    unit::U
    detrend::Symbol
    baseline::Vector{Float64}

    function AnnMaxRecord(
        station::Station,
        years::Vector{Int},
        levels::Vector{Float64},
        unit::U,
        detrend::Symbol,
        baseline::Vector{Float64},
    ) where {U<:Units}
        length(years) == length(levels) == length(baseline) || throw(
            DimensionMismatch("years, levels and baseline must be the same length"),
        )
        detrend in DETREND_METHODS ||
            throw(ArgumentError("detrend must be one of $DETREND_METHODS, got :$detrend"))
        return new{U}(station, years, levels, unit, detrend, baseline)
    end
end

const TideRecord = Union{WaterLevelRecord,AnnMaxRecord}

"""
    waterlevels(record) -> Vector{<:Length}

Every level in the record, with its unit attached.

`ustrip.(waterlevels(record))` gives the bare numbers back, and
`ustrip.(u"ft", waterlevels(record))` converts on the way.
"""
waterlevels(record::TideRecord) = record.levels .* record.unit

"""
    obstimes(record) -> Vector{DateTime}

When each reading was taken.
"""
obstimes(record::WaterLevelRecord) = record.times

"""
    obsyears(record) -> Vector{Int}

The year of each annual maximum.
"""
obsyears(record::AnnMaxRecord) = record.years

"""
    baseline(record) -> Vector{<:Length}

What detrending subtracted from each year, with units. A `:none` record has a
baseline of zeros.
"""
baseline(record::AnnMaxRecord) = record.baseline .* record.unit

# Both records are a sequence of readings first and anything else second, so the
# iteration interface is what makes `length`, `collect` and `for` work. An
# element comes back as a named tuple carrying its unit, which is what a reader
# wants; the bare vectors stay available through `waterlevels` and `ustrip`.
Base.length(record::TideRecord) = length(record.levels)
Base.firstindex(::TideRecord) = 1
Base.lastindex(record::TideRecord) = length(record)
Base.iterate(record::TideRecord, state=1) =
    state > length(record) ? nothing : (record[state], state + 1)

Base.getindex(record::WaterLevelRecord, i::Integer) =
    (time=record.times[i], level=record.levels[i] * record.unit)
Base.getindex(record::AnnMaxRecord, i::Integer) =
    (year=record.years[i], level=record.levels[i] * record.unit)

Base.eltype(::Type{<:WaterLevelRecord{U}}) where {U} =
    @NamedTuple{time::DateTime, level::Quantity{Float64,dimension(u"m"),U}}
Base.eltype(::Type{<:AnnMaxRecord{U}}) where {U} =
    @NamedTuple{year::Int, level::Quantity{Float64,dimension(u"m"),U}}

function Base.show(io::IO, ::MIME"text/plain", record::WaterLevelRecord)
    println(io, "WaterLevelRecord: $(length(record)) readings in $(record.unit)")
    println(io, "  station: ", record.station)
    if !isempty(record)
        println(io, "  period:  ", minimum(record.times), " to ", maximum(record.times))
        print(
            io,
            "  levels:  ",
            minimum(record.levels),
            " to ",
            maximum(record.levels),
            " ",
            record.unit,
        )
    end
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", record::AnnMaxRecord)
    println(
        io,
        "AnnMaxRecord: $(length(record)) years in $(record.unit), " *
        "detrended by :$(record.detrend)",
    )
    println(io, "  station: ", record.station)
    if !isempty(record)
        println(io, "  period:  ", minimum(record.years), " to ", maximum(record.years))
        print(
            io,
            "  levels:  ",
            minimum(record.levels),
            " to ",
            maximum(record.levels),
            " ",
            record.unit,
        )
    end
    return nothing
end

"""
    AnnMaxRecord(record; detrend=:linear, units=u"ft", min_readings=6_000)

Take the annual maxima from a raw water level record and remove the sea-level
trend.

A year's maximum is the largest reading that falls in it. `detrend` is one of
$DETREND_METHODS and is described in [`detrend_baseline`](@ref); `:msl` uses each
year's own mean reading as the baseline, which is what the raw record makes
available.

    AnnMaxRecord(load_water_level("8638610"); detrend=:msl)

`min_readings` sets how many readings a year needs before it is kept, which is
what drops the incomplete year at the end of a record and any gap in the middle.
Its default assumes hourly data, where a full year holds 8,760 readings, so
lower it for a coarser record.
"""
function AnnMaxRecord(
    record::WaterLevelRecord;
    detrend::Symbol=:linear,
    units::Units=u"ft",
    min_readings::Integer=6_000,
)
    detrend in DETREND_METHODS ||
        throw(ArgumentError("detrend must be one of $DETREND_METHODS, got :$detrend"))

    rows = Dict{Int,Vector{Float64}}()
    for (t, level) in zip(record.times, record.levels)
        push!(get!(rows, year(t), Float64[]), level)
    end

    kept = sort([y for (y, vals) in rows if length(vals) >= min_readings])
    isempty(kept) && throw(
        ArgumentError(
            "no year in this record has $min_readings readings; the record holds " *
            "$(length(record)) readings over $(length(rows)) years, so lower " *
            "min_readings if it is not hourly",
        ),
    )

    # One conversion factor for the whole record, since every level shares a unit.
    factor = ustrip(uconvert(units, oneunit(1.0 * record.unit)))
    annmax = [maximum(rows[y]) * factor for y in kept]
    # `:msl` needs a mean sea level per year, which the raw readings give
    # directly: the mean of every reading in a year is that year's mean level.
    msl = [mean(rows[y]) * factor for y in kept]

    trend = detrend_baseline(kept, annmax, msl, detrend)
    detrended = annmax .- trend .+ recentre(trend)

    return AnnMaxRecord(record.station, kept, detrended, units, detrend, trend)
end

"""
    DataFrame(record; keep_units=false)

A record as a table, one row per reading.

`keep_units=true` gives a level column of `Unitful` quantities; the default
gives bare numbers, which is what a fit and a plot both want. The unit is in the
column name either way.

    DataFrame(annmax)                      # year, level_ft, baseline_ft
    DataFrame(annmax; keep_units=true)     # level carries u"ft"
"""
function DataFrames.DataFrame(record::WaterLevelRecord; keep_units::Bool=false)
    level = keep_units ? waterlevels(record) : record.levels
    return DataFrame(; time=record.times, Symbol(level_column(record)) => level)
end

function DataFrames.DataFrame(record::AnnMaxRecord; keep_units::Bool=false)
    level = keep_units ? waterlevels(record) : record.levels
    trend = keep_units ? baseline(record) : record.baseline
    return DataFrame(;
        year=record.years,
        Symbol(level_column(record)) => level,
        Symbol("baseline_$(record.unit)") => trend,
    )
end

level_column(record::TideRecord) = "level_$(record.unit)"
