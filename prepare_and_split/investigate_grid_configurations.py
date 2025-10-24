from time import time
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import logging
import pandas as pd
import numpy as np

import config
from utils.preparation import prepare_database, grid_data_as_df


# # Quick global imputation
# from sklearn.impute import SimpleImputer
# from utils.plotting import plot_each_depth_level, animate_depth_panels
# from utils.gridding import df_to_gridded_da
# df = pd.read_csv("output/gridding/df_20y_global.csv")
# df["DATEANDTIME"] = pd.to_datetime(df["DATEANDTIME"]).dt.year
# x, y, scaler_dicts = preprocess(df, coord_names=config.coordinates, parameter_names=config.parameters)
# cols = ["LATITUDE", "cLONGITUDE", "sLONGITUDE", "LEV_M", "DATEANDTIME"] + config.parameters
# imputer = SimpleImputer()
# df_imputed = pd.DataFrame(imputer.fit_transform(np.concat([x, y], axis=1)), columns=cols)
# df_imputed[config.coordinates] = df[config.coordinates]
#
# param = "P_TEMPERATURE"
# da = df_to_gridded_da(df=df_imputed, value_col=param)
# animate_depth_panels(da, save_as=f"output/{param}_knn_imputation.mp4")


def test_grids():
    return

# prepare_database(parameters=config.parameters,
#                  quality_flags=config.quality_flags,
#                  temperature_to_potential=True,
#                  source_db_path=config.source_db_path,
#                  dest_db_path=config.dest_db_path)

# Check missingness for various grid configurations
res = []
for grid_id, grid_config in config.grid_configs.items():
    st = time()
    if grid_id == "20y_global":
        print(grid_id)
        df, table_name = grid_data_as_df(db_path=config.dest_db_path, grid_config=grid_config,
                                         bathymetry_path=config.bathymetry_path,
                                         parameters=config.parameters,
                                         output_dir=config.output_dir)

        # Store
        df.to_csv(f"{config.output_dir_gridding}df_{grid_id}.csv", index=False)

        # Plot missingness per time step
        total_grid_cells_per_time = df.groupby('DATEANDTIME')["LATITUDE"].count()
        missing_counts = df.groupby('DATEANDTIME')[config.parameters].apply(lambda x: x.isna().sum())
        missing_fraction = missing_counts.div(total_grid_cells_per_time, axis=0) * 100

        plt.figure()
        for feat in config.parameters:
            plt.plot(missing_fraction.index, missing_fraction[feat], label=feat)

        plt.ylabel("Missigness [%]")
        plt.xlabel("Time")
        plt.title(f"Time Step: {grid_id}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(config.output_dir_preliminary + f"/missingness_{grid_id}.png", dpi=1000)
        plt.show()

        print(f"Grid {grid_id} took {time() - st} seconds.")

end = time()
logging.info(f"Database preparation and gridding took {end - st} seconds.")
