import pandas as pd
import matplotlib.pyplot as plt

csv_path = "mpi_saale_2021b.csv"
data_frame = pd.read_csv(csv_path)
print(data_frame.columns)

# Index(['Date Time', 'p (mbar)', 'T (degC)', 'rh (%)', 'sh (g/kg)', 'Tpot (K)',
#        'Tdew (degC)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)',
#        'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'wd (deg)', 'rain (mm)',
#        'SWDR (W/m**2)', 'SDUR (s)', 'TRAD (degC)', 'Rn (W/m**2)',
#        'ST002 (degC)', 'ST004 (degC)', 'ST008 (degC)', 'ST016 (degC)',
#        'ST032 (degC)', 'ST064 (degC)', 'ST128 (degC)', 'SM008 (%)',
#        'SM016 (%)', 'SM032 (%)', 'SM064 (%)', 'SM128 (%)'], dtype='object')

print(data_frame['T (degC)'])
print(data_frame['Date Time'])


#features = pandas.concat([temperature, pressure, relative_humidity, vapor_pressure, wind_speed, airtight], axis=1)
#y_train = features.iloc[start:end][[0]] #温度数据，即预测温度

# # 提取温度数据
# temperature_data = data_frame['T (degC)']

# # 创建日期时间索引
# date_index = pd.to_datetime(data_frame['Date Time'])

# # 创建温度数据的图表
# plt.figure(figsize=(10, 6))
# plt.plot(date_index[:100], temperature_data[:100])
# plt.title('Temperature Data')
# plt.xlabel('Time')
# plt.ylabel('Temperature (Celsius)')
# #plt.grid(True)
# plt.show()