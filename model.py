import pandas as pd

def analyze_customer(df, customer_id):

    # Filter customer data
    customer_data = df[df["CustomerID"] == customer_id]

    if customer_data.empty:
        return None

    # Ensure InvoiceDate is datetime
    customer_data["InvoiceDate"] = pd.to_datetime(customer_data["InvoiceDate"])

    # Calculations
    total_spent = (customer_data["Quantity"] * customer_data["UnitPrice"]).sum()
    total_orders = customer_data["InvoiceNo"].nunique()
    avg_order_value = round(total_spent / total_orders, 2)

    favorite_country = customer_data["Country"].mode()[0]

    sales_velocity = customer_data["Quantity"].sum()

    customer_type = "High Value" if total_spent > 5000 else "Regular"

    # ✅ Velocity trend over time (THIS WAS THE IMPORTANT PART)
    daily_velocity = (
        customer_data
        .groupby(customer_data["InvoiceDate"].dt.date)["Quantity"]
        .sum()
        .reset_index()
    )

    daily_velocity.columns = ["date", "velocity"]

    # ✅ RETURN MUST BE INSIDE THE FUNCTION
    return {
        "customer_id": customer_id,
        "total_spent": round(total_spent, 2),
        "total_orders": total_orders,
        "avg_order_value": avg_order_value,
        "favorite_country": favorite_country,
        "customer_type": customer_type,
        "sales_velocity": int(sales_velocity),
        "velocity_trend": daily_velocity.to_dict(orient="records")
    }