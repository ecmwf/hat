`hat-hydrostats` documentation
==============================

Command line tool to calculate hydrological statistics on timeseries. 

How to use
-----
To run this analysis the following are required:

- `--functions` = names of statistical function(s)
- `--sims` = filepath to simulation file
- `--obs` = filepath to observation file

For example

`hydrostats --functions kge --sims $SIMS --obs $OBS`

These are the currently supported functions:

- apb
- apb2
- bias
- br
- correlation
- kge
- index_agreement
- mae
- ns
- nslog
- pc_bias
- pc_bias2
- rmse
- rsr
- vr

You can calculate more than one function at once using commas with the `--functions` option

`hat-hydrostats --functions kge, rmse, mae, correlation --sims $SIMS --obs $OBS`

(Optionally) define the minimum percentage of observations required for timeseries to be valid using the `--obs_threshold` option (default is 80%)

`hat-hydrostats --functions kge --sims $SIMS --obs $OBS --obs_threshold 70`