# Column interface, so a table can be named the way Extremes.jl names one.
#
# Through Tables.jl rather than DataFrames, so a DataFrame, a NamedTuple of
# vectors, and a CSV.File all work and the package takes no heavy dependency to
# get there.

_column(cols, name) = Tables.getcolumn(cols, name)
_columns(cols, names) = isempty(names) ? nothing : [_column(cols, n) for n in names]

"""
    gevfit(table, datacol; locationcovid, logscalecovid, shapecovid, kwargs...)

Fit from a column of any Tables.jl table, naming covariate columns by symbol.

    gevfit(df, :lsl; locationcovid = [:year])
"""
function gevfit(
    table,
    datacol::Symbol;
    locationcovid=Symbol[],
    logscalecovid=Symbol[],
    shapecovid=Symbol[],
    kwargs...,
)
    cols = Tables.columns(table)
    return gevfit(
        _column(cols, datacol);
        locationcov=_columns(cols, locationcovid),
        logscalecov=_columns(cols, logscalecovid),
        shapecov=_columns(cols, shapecovid),
        kwargs...,
    )
end

"""
    gpfit(table, datacol, threshold; logscalecovid, shapecovid, kwargs...)

Fit a peaks over threshold model from a column of any Tables.jl table.
"""
function gpfit(
    table,
    datacol::Symbol,
    threshold::Real;
    logscalecovid=Symbol[],
    shapecovid=Symbol[],
    kwargs...,
)
    cols = Tables.columns(table)
    return gpfit(
        _column(cols, datacol),
        threshold;
        logscalecov=_columns(cols, logscalecovid),
        shapecov=_columns(cols, shapecovid),
        kwargs...,
    )
end

"""
    gevfitbayes(table, datacol; locationcovid, logscalecovid, shapecovid, kwargs...)

Sample the posterior from a column of any Tables.jl table.
"""
function gevfitbayes(
    table,
    datacol::Symbol;
    locationcovid=Symbol[],
    logscalecovid=Symbol[],
    shapecovid=Symbol[],
    kwargs...,
)
    cols = Tables.columns(table)
    return gevfitbayes(
        _column(cols, datacol);
        locationcov=_columns(cols, locationcovid),
        logscalecov=_columns(cols, logscalecovid),
        shapecov=_columns(cols, shapecovid),
        kwargs...,
    )
end

"""
    gpfitbayes(table, datacol, threshold; logscalecovid, shapecovid, kwargs...)

Sample the posterior of a peaks over threshold fit from a table column.
"""
function gpfitbayes(
    table,
    datacol::Symbol,
    threshold::Real;
    logscalecovid=Symbol[],
    shapecovid=Symbol[],
    kwargs...,
)
    cols = Tables.columns(table)
    return gpfitbayes(
        _column(cols, datacol),
        threshold;
        logscalecov=_columns(cols, logscalecovid),
        shapecov=_columns(cols, shapecovid),
        kwargs...,
    )
end
