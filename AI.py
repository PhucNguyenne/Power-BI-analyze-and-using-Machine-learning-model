import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Đọc dữ liệu
df = pd.read_csv('data.csv')
df['incident_date'] = pd.to_datetime(df['incident_date'])

# Xử lý dữ liệu: điền giá trị thiếu
for col in df.columns[1:]:
    df[col] = df[col].fillna(df[col].mode()[0])
        
# Tạo biến đặc trưng cho ngày trong tháng
df['day'] = df['incident_date'].dt.day
df['month'] = df['incident_date'].dt.month
df['year'] = df['incident_date'].dt.year

# Mã hóa các cột phân loại thành các giá trị số
df = pd.get_dummies(df, drop_first=True)

# Tách dữ liệu thành đặc trưng và mục tiêu
X = df.drop(columns=['total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim', 'incident_date'])
y_total = df['total_claim_amount']
y_injury = df['injury_claim']
y_property = df['property_claim']
y_vehicle = df['vehicle_claim']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train_total, y_test_total = train_test_split(X, y_total, test_size=0.2, random_state=42)
X_train, X_test, y_train_injury, y_test_injury = train_test_split(X, y_injury, test_size=0.2, random_state=42)
X_train, X_test, y_train_property, y_test_property = train_test_split(X, y_property, test_size=0.2, random_state=42)
X_train, X_test, y_train_vehicle, y_test_vehicle = train_test_split(X, y_vehicle, test_size=0.2, random_state=42)

# Tạo mô hình Random Forest
model_total = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=2)
model_injury = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=2)
model_property = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=2)
model_vehicle = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=2)

# Huấn luyện mô hình
model_total.fit(X_train, y_train_total)
model_injury.fit(X_train, y_train_injury)
model_property.fit(X_train, y_train_property)
model_vehicle.fit(X_train, y_train_vehicle)

# Dự đoán trên tập kiểm tra
predicted_total = model_total.predict(X_test)
predicted_injury = model_injury.predict(X_test)
predicted_property = model_property.predict(X_test)
predicted_vehicle = model_vehicle.predict(X_test)

# Tính MAE và MSE
print(f'Total Claim - MAE: {mean_absolute_error(y_test_total, predicted_total)} MSE: {mean_squared_error(y_test_total, predicted_total)}')
print(f'Injury Claim - MAE: {mean_absolute_error(y_test_injury, predicted_injury)} MSE: {mean_squared_error(y_test_injury, predicted_injury)}')
print(f'Property Claim - MAE: {mean_absolute_error(y_test_property, predicted_property)} MSE: {mean_squared_error(y_test_property, predicted_property)}')
print(f'Vehicle Claim - MAE: {mean_absolute_error(y_test_vehicle, predicted_vehicle)} MSE: {mean_squared_error(y_test_vehicle, predicted_vehicle)}')

# Dự đoán cho 30 ngày tiếp theo
future_dates = pd.date_range(start='2024-11-01', periods=30, freq='D')
X_future = pd.DataFrame({
    'day': range(1, 31),  # Ngày từ 1 đến 30
    'month': [11] * 30,   # Tháng 11
    'year': [2024] * 30    # Năm 2024
})

# Dự đoán cho 30 ngày tiếp theo
future_dates = pd.date_range(start='2024-11-01', periods=30, freq='D')
X_future = pd.DataFrame({
    'day': range(1, 31),  # Ngày từ 1 đến 30
    'month': [11] * 30,   # Tháng 11
    'year': [2024] * 30    # Năm 2024
})

# Mã hóa các cột phân loại trong X_future
X_future = pd.get_dummies(X_future, drop_first=True)

# Đảm bảo rằng X_future có cùng cột với X_train
X_future = X_future.reindex(columns=X.columns, fill_value=0)

# Dự đoán
predicted_total_claim = model_total.predict(X_future)
predicted_injury_claim = model_injury.predict(X_future)
predicted_property_claim = model_property.predict(X_future)
predicted_vehicle_claim = model_vehicle.predict(X_future)

# Thêm nhiễu ngẫu nhiên vào giá trị dự đoán
noise_total = np.random.normal(0, 1000, size=predicted_total_claim.shape)
predicted_total_claim += noise_total

# Lưu kết quả vào file CSV
predictions_df = pd.DataFrame({
    'Date': future_dates.day,  # Chỉ hiển thị ngày 1-30
    'Total Claim': (predicted_injury_claim + predicted_property_claim + predicted_vehicle_claim),
    'Injury Claim': predicted_injury_claim,
    'Property Claim': predicted_property_claim,
    'Vehicle Claim': predicted_vehicle_claim
})

predictions_df.to_csv('predicted_claims.csv', index=False)

# Vẽ biểu đồ
plt.figure(figsize=(12, 6))
plt.plot(predictions_df['Date'], predictions_df['Total Claim'], label='Total Claim', marker='o')
# plt.plot(predictions_df['Date'], predictions_df['Injury Claim'], label='Injury Claim', marker='o')
# plt.plot(predictions_df['Date'], predictions_df['Property Claim'], label='Property Claim', marker='o')
# plt.plot(predictions_df['Date'], predictions_df['Vehicle Claim'], label='Vehicle Claim', marker='o')
plt.title('Predicted Claims trong tháng tiếp theo')
plt.xlabel('Date ')
plt.ylabel('Claim Amount')
plt.xticks(predictions_df['Date'])  # Đảm bảo hiển thị các ngày
plt.legend()
plt.grid()
plt.show()
