from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import DateField
from wtforms.validators import DataRequired
import pandas as pd
from prophet import Prophet
import yfinance as yf
from datetime import datetime, timedelta

app = Flask(__name__,template_folder='template')
app.config['SECRET_KEY'] = 'your_secret_key'

class DateSelectionForm(FlaskForm):
    selected_date = DateField('Select a Date', validators=[DataRequired()])

@app.route('/', methods=['GET', 'POST'])
def home():
    form = DateSelectionForm()

    # Default values for the date range
    start_date = "2019-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')

    if form.validate_on_submit():
        # Update the end_date based on the user-selected date
        end_date = form.selected_date.data.strftime('%Y-%m-%d')

    # Load historical Bitcoin price data
    symbol = "BTC-USD"
    bitcoin_data = yf.download(symbol, start=start_date, end=end_date)

    # Prepare the data for Prophet
    df = pd.DataFrame()
    df['ds'] = bitcoin_data.index
    df['y'] = bitcoin_data['Close'].values

    # Feature Engineering
    df['returns'] = df['y'].pct_change()
    df['ma7'] = df['y'].rolling(window=7).mean()
    df['ma30'] = df['y'].rolling(window=30).mean()
    df['volatility'] = df['returns'].rolling(window=7).std()
    df = df.dropna()

    # Instantiate the Prophet model
    model3 = Prophet(changepoint_prior_scale=0.05, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)

    # Add additional regressors
    model3.add_regressor('ma7')
    model3.add_regressor('ma30')
    model3.add_regressor('volatility')

    # Fit the model
    model3.fit(df)

    # Create a dataframe for the selected day
    future = pd.DataFrame({'ds': [end_date]})

    # Add regressor values for the selected day
    future['ma7'] = df['ma7'].values[-1]
    future['ma30'] = df['ma30'].values[-1]
    future['volatility'] = df['volatility'].values[-1]

    # Make predictions for the selected day
    forecast = model3.predict(future)

    # Extract the predicted closing price for the selected day
    predicted_price = forecast['yhat'].iloc[0]

    return render_template('index.html', form=form, predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
