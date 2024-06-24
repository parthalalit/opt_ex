from michalis import michaelis_menten

def objective_function(x, products, parameters):

    sum_orders = 0.0

    # Unpack parameters for each product and calculate expected orders
    for product, budget in zip(products, x, strict=False):
        L, k = parameters[product]
        sum_orders += michaelis_menten(budget, L, k)

    return -1 * sum_orders