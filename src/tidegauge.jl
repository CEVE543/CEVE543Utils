# Water levels from NOAA CO-OPS, and the sea-level trend removed from them.
#
# The `hourly_height` product reports one water level per hour and reaches back
# to the start of a long gauge's record, so one download serves both the annual
# maxima and the peaks-over-threshold work that uses the same readings.
#
# Station ids are listed at https://tidesandcurrents.noaa.gov/stations.html.

using CSV: CSV
using Dates: DateTime, @dateformat_str, year, today
using Downloads: download
using Unitful: @u_str

const COOPS_API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
const COOPS_STATIONS = "https://tidesandcurrents.noaa.gov/stations.html"
# The detrended series sits at the mean level of the last this many years.
const REF_WINDOW = 5
const DETREND_METHODS = (:linear, :msl, :none)

"""
    load_water_level(station; kwargs...) -> WaterLevelRecord

Download hourly water levels from NOAA CO-OPS tide gauge `station`.

Returns a [`WaterLevelRecord`](@ref) carrying the station and one reading per
hour, in metres above the requested datum.

    record = load_water_level("8638610")          # Sewells Point, VA
    annmax = AnnMaxRecord(record; detrend=:msl)

Station ids are listed at $COOPS_STATIONS.

A long record is a large download: the API serves one year per request, so a
century of hourly readings is around 80 requests and 30 MB, and takes a minute
or two. It is cached, so that cost is paid once.

# Keywords

- `cache`: path to a CSV to read instead of the network, and to write after a
  download. `nothing` disables caching. Defaults to `"<station>-hourly.csv"`,
  which resolves against the working directory, so pass an absolute path built
  with `@__DIR__` to get the same file from a notebook and from the REPL.
- `first_year`, `last_year`: the period to request.
- `datum`: the vertical reference, `"MSL"` by default.
- `name`: how to label the station in output.
"""
function load_water_level(
    station::AbstractString;
    cache=default_cache(station),
    first_year::Integer=1928,
    last_year::Integer=year(today()),
    datum::AbstractString="MSL",
    name::AbstractString="",
)
    meta = Station(station; name=name, datum=datum)

    if cache !== nothing && isfile(cache)
        # `types` is what keeps the timestamp a `DateTime`; CSV.jl otherwise
        # parses it to its own timestamp type, which the record rejects.
        cached = CSV.File(cache; types=Dict(:time => DateTime, :level_m => Float64))
        has_columns(cached, (:time, :level_m)) || throw(
            ArgumentError(
                "the cache at $cache is missing a time or level_m column; " *
                "delete it to download the station again",
            ),
        )
        return WaterLevelRecord(
            meta, collect(cached.time), collect(cached.level_m), u"m"
        )
    end

    times, levels = DateTime[], Float64[]
    for yr in first_year:last_year
        append_year!(times, levels, station, yr, datum)
    end
    isempty(levels) && throw(
        ArgumentError(
            "station $station returned no readings for $first_year-$last_year; " *
            "check the id at $COOPS_STATIONS",
        ),
    )

    order = sortperm(times)
    times, levels = times[order], levels[order]
    cache === nothing || CSV.write(cache, (time=times, level_m=levels))
    # The API serves metric, which is why the record is in metres whatever the
    # annual maxima are later converted to.
    return WaterLevelRecord(meta, times, levels, u"m")
end

default_cache(station::AbstractString) = "$station-hourly.csv"

has_columns(file, wanted) = all(c -> c in propertynames(file), wanted)

"""
    append_year!(times, levels, station, yr, datum)

Download one year of hourly readings and append them. A year the gauge did not
report adds nothing rather than raising, since a long record routinely has gaps.
"""
function append_year!(
    times::Vector{DateTime},
    levels::Vector{Float64},
    station::AbstractString,
    yr::Integer,
    datum::AbstractString,
)
    url = (
        "$COOPS_API?product=hourly_height&application=NOS.COOPS.TAC.WL" *
        "&begin_date=$(yr)0101&end_date=$(yr)1231" *
        "&datum=$datum&station=$station&time_zone=GMT&units=metric&format=csv"
    )
    # A wrong station id is answered with HTTP 400 rather than an empty CSV, so
    # the download is where a typo has to be caught. A year outside the gauge's
    # record answers 200 with a one-line body, which parses to nothing.
    body = try
        sprint(io -> download(url, io))
    catch err
        throw(
            ArgumentError(
                "could not download station $station for $yr; " *
                "check the id at $COOPS_STATIONS ($err)",
            ),
        )
    end

    # `normalizenames` is what strips the leading spaces the product puts in its
    # header, so that `row.Water_Level` resolves. A year outside the gauge's
    # record answers with a header and nothing else, which has no columns.
    table = CSV.File(IOBuffer(body); normalizenames=true)
    has_columns(table, (:Date_Time, :Water_Level)) || return nothing

    # A gap in the record arrives as a blank cell, which CSV.jl gives as
    # `missing`. Anything else that fails to convert is a change in the
    # product's schema and should be raised rather than swallowed.
    for row in table
        (ismissing(row.Date_Time) || ismissing(row.Water_Level)) && continue
        push!(times, DateTime(row.Date_Time, dateformat"yyyy-mm-dd HH:MM"))
        push!(levels, Float64(row.Water_Level))
    end
    return nothing
end

"""
    detrend_baseline(years, annmax, msl, method) -> Vector{Float64}

What to subtract from each year's maximum to remove the sea-level trend.

`method` is one of:

- `:linear` fits a straight line to the annual maxima by ordinary least squares.
  It assumes the trend in the maxima is linear in time and estimates it from the
  maxima alone, which are noisy.
- `:msl` uses each year's own mean sea level. The baseline is measured rather
  than fitted, and it follows whatever the sea actually did, including the
  decadal swings a straight line cannot see. This leaves the storm surge on top
  of the tide, which is the quantity a GEV is being asked about.
- `:none` subtracts nothing, which is what a non-stationary model wants.

The caller re-centres the result with [`recentre`](@ref), so the detrended
series is expressed at recent sea level rather than at zero.
"""
function detrend_baseline(
    years::AbstractVector{<:Real},
    annmax::AbstractVector{<:Real},
    msl::AbstractVector{<:Real},
    method::Symbol,
)
    method === :none && return zeros(float(eltype(annmax)), length(annmax))
    if method === :linear
        slope, intercept = ols(years, annmax)
        return slope .* years .+ intercept
    elseif method === :msl
        return collect(float.(msl))
    end
    throw(ArgumentError("detrend must be one of $DETREND_METHODS, got :$method"))
end

"""
    recentre(baseline) -> Float64

The level a detrended series is expressed at: the mean of the baseline over the
last $REF_WINDOW years, so the result reads as present-day sea level rather than
as a departure from zero.
"""
recentre(baseline::AbstractVector{<:Real}) =
    mean(baseline[max(end - REF_WINDOW + 1, 1):end])

"""
    ols(x, y) -> (slope, intercept)

Least squares fit of a straight line.
"""
function ols(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    mean_x, mean_y = mean(x), mean(y)
    slope = sum((x .- mean_x) .* (y .- mean_y)) / sum((x .- mean_x) .^ 2)
    return slope, mean_y - slope * mean_x
end
