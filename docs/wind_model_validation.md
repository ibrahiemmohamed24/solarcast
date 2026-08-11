# Wind model: method and checks

This note explains how the weather data is turned into an estimate of power output, what we tested, and what the results actually tell us.

## How the model works

NASA POWER gives us wind speed at 10 m and 50 m, along with temperature and pressure. The wind farms use hub heights between 63 m and 110 m, so the model has to bridge that gap before it can estimate output.

### Wind shear

Wind speed changes with height, and the change depends on the ground and local conditions. Instead of using the usual fixed value of 0.14, we calculate the shear exponent hour by hour from the two NASA heights. At the Gulf of Suez sites, the values usually fall between 0.145 and 0.175. A published mast measurement from Ras Ghareb reported 0.18, which is the closest outside check we have.

### Air density

Air density affects how much power is available in the wind. Hot desert air is less dense, so using the standard value without correction would push the estimate too high. We calculate density from temperature and pressure using the ideal gas law. Across the sites, it comes out between 1.15 and 1.18 kg/m3, around five percent below the standard 1.225 used for published power curves.

### Density correction

The IEC 61400-12 method applies the density correction to wind speed rather than directly to power. It uses a cube-root correction for pitch-regulated turbines and a linear correction for stall-regulated turbines.

## Power curves

Manufacturers usually publish their curves as images. None of the eight turbines in this project is available in the windpowerlib library. The library contains 64 turbines, mostly German models, with no Gamesa, Nordex N43, Envision, or Goldwind models.

### The first method did not work

The first attempt used a cubic curve between cut-in speed and rated speed. When we tested it against the measured V90/2000 curve, it was about 30 percent low through the ramp and 28 percent low at 10 m/s, which is one of the most important speeds for these sites.

The reason is simple. The cubic law describes the energy in the wind, but not how a real turbine takes that energy. Real turbines run close to their best efficiency at lower speeds, then reduce that efficiency as they approach rated output.

### The method used now

We start with measured turbine curves and convert them into power-coefficient curves. We then place them on a common speed scale where cut-in is 0 and rated speed is 1. The curves are averaged and then scaled back to the rotor size and rated power of the target turbine.

In short, the curve shape comes from measured machines, while the size and rating come from the target turbine specifications.

### Choosing the reference curves

We tested two options: average the whole library, or use the five closest references based on rated-to-cut-in ratio and specific power. The test rules, main metric, holdout method, practical margin, and bootstrap setup were written down before the run. The code is in src/models/reference_selection_experiment.py.

For the main test, the whole turbine family was removed from the reference library. Removing only the target turbine would give the nearest-reference method an easy advantage whenever a closely related model was still available.

The library has 64 turbines in 39 families. Forty-one turbines belong to families with more than one member. None of the Egyptian turbines has a related family in the library, so holding out the full family is the better match for the real job.

The five-reference method improved the median by 0.61 percentage points. The 95 percent confidence interval ran from minus 0.53 to plus 1.06 percentage points. That does not clear the 0.3-point practical margin that was set before the test. It also improved the median while making the p90 result worse.

The decision is to average the whole library.

This does not mean reference selection has no value. It means the test on these 39 families did not show a large and reliable enough benefit to justify the extra choice.

With full-family holdout and Gulf of Suez wind conditions, the absolute AEP error for the chosen method had a median of 1.95 percent, a mean of 2.71 percent, and a p90 of 5.63 percent. Under a milder wind distribution, the median was 2.16 percent.

Errors at individual wind speeds are larger than the final AEP error because some high and low errors cancel when the curve is integrated over the wind distribution.

There are limits to this test. Most of the library contains modern European turbines. The results may be too optimistic for 1990s stall-regulated machines and recent Chinese turbines, since neither group is properly represented. Eleven families have a same-specific-power match from another manufacturer. The family holdout does not remove those matches. We kept them because they represent coverage of the curve space rather than a repeated turbine identity.

## Finding a bad source value

An 80 m rotor producing 2000 kW at 12 m/s would need a power coefficient of 0.388. Only four turbines in the library reach that level at their own rated point, while the library median is 0.288. That made the published rated speed for the Gamesa G80 physically doubtful.

The check_rated_speed check now compares every published rated speed with the lowest speed that is physically possible. Across the library, the median difference is minus 0.1 m/s. Only three of the 64 turbines fall more than 1 m/s below the physical limit, so the check catches unusual values without flagging everything. Of the eight project turbines, only the G80 was flagged.

Its rated speed was corrected to 14.0 m/s. That is the value published by Vestas for the V80-2.0, which has the same rating, rotor size, and specific power. The change is based on physics and a comparable machine, not on adjusting the model until it matches reported generation.

## Keeping reported generation out of tuning

Reported generation is the only independent result we can use to check the model, so it cannot also be used to tune the inputs.

We follow three rules.

1. Every corrected value records the old value, the new value, the source, and the reason for the change.

2. A correction can come from physics or from a genuinely comparable turbine. It cannot come from moving a value until the output matches reported generation.

3. The number of adjustable values stays small. With enough adjustable numbers, almost any target can be matched and the comparison stops meaning anything.

This is why the hub height at Gabal El Zeit is not chosen according to whichever value gets closest to 2400 GWh. The available sources point to 67 m, while 78 m happens to give the closest model result. Both values are recorded, but the reported output is not used to choose between them.

## Checks with deliberately bad inputs

A close result only matters if the model becomes worse when it is given bad data. We set the limits before running the checks. An error below 15 percent with broken input is weak evidence. An error above 30 percent shows that the model is reacting to the input data.

| Site | Baseline | Time shuffled | Random wind | Worst site swap |
|---|---|---|---|---|
| Zafarana | -5.5% | -5.4% | +18.0% | +88.6% |
| Gabal El Zeit | -11.1% | -11.2% | -16.2% | -52.9% |
| West Bakr | +13.2% | +13.0% | +0.2% | -30.2% |
| Amunet | -39.1% | -39.1% | -34.2% | -4.4% |

Swapping sites usually causes a large drop in accuracy, from about 30 to 89 percent. This is the strongest check here because it shows that the model reacts to the wind data from each site.

Shuffling time makes almost no difference. That is expected because hourly power is being added up and the order of the hours does not change the total. This model estimates wind resources; it is not a forecasting model and does not yet contain a time-based component.

Random wind with the correct mean produces errors ranging from 0.2 to 34 percent. West Bakr matching within 0.2 percent using invented data is a useful warning: one close site result does not prove much on its own. The broader evidence comes from several farms with different sizes, ages, and turbine types landing in a reasonable range.

Some site swaps make no difference. This happens for Zafarana and Amunet, and for Gabal El Zeit and West Bakr. Each pair falls on the same NASA POWER grid cell and receives identical or nearly identical weather series. This was found at the start of the work and recorded in config/grid_collisions.json. The bad-input test found the same issue again on its own.

## Current results

| Site | Modelled | Reported | Source of reported | Error |
|---|---|---|---|---|
| Zafarana | 892 GWh/yr | 944 GWh/yr | NREA, 2008-09 | -5.5% |
| Gabal El Zeit | 2133 GWh/yr | 2400 GWh/yr | Operator | -11.1% |
| West Bakr | 1132 GWh/yr | 1000 GWh/yr | Siemens Gamesa | +13.2% |
| Amunet | 1523 GWh/yr | 2500 GWh/yr | Operator | -39.1% |

Zafarana is the strongest comparison. It includes three turbine types, none with a published curve, and the reported figure comes from the national authority.

Amunet is the clear outlier. There are two likely reasons.

First, NASA POWER uses MERRA-2 wind data at a resolution of roughly 50 km. Published work covering 23 countries found that reanalysis data can underestimate output by around 30 percent in Mediterranean conditions. The Gulf of Suez coast is a narrow strip between the sea and the escarpment, so one large grid cell smooths out important local wind conditions.

Second, the 2500 GWh figure appears only in a press release. It would require a net capacity factor of 57 percent, higher than every documented onshore record we found. The highest independently metered Egyptian value is around 54.7 percent net at Ras Ghareb in 2022, based on carbon-credit documents rather than marketing material. A more believable range for Amunet is 2100 to 2280 GWh.

## Work still not done

The model has not yet been bias-corrected against the eleven Gulf of Suez masts in the Wind Atlas for Egypt. Their Weibull values are published, but their exact coordinates and measurement periods are not. Without those details, NASA POWER cannot be sampled at the same place and over the same period for a fair comparison.

The model also has no forecasting or other time-based component yet.
