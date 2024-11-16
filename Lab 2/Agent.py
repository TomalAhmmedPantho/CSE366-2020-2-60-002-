import random
import matplotlib.pyplot as plt


# Utility function for random sampling from a distribution
def select_from_dist(item_prob_dist):
    ranreal = random.random()
    for item, prob in item_prob_dist.items():
        if ranreal < prob:
            return item
        ranreal -= prob
    raise RuntimeError(f"{item_prob_dist} is not a valid probability distribution")


# Plotting Class
class PlotHistory:
    def __init__(self, agent, environment):
        self.agent = agent
        self.environment = environment

    def plot_history(self):
        plt.figure(figsize=(12, 8))

        # Plot price history
        plt.plot(
            self.environment.price_history,
            label="Price",
            color="purple",
            linewidth=2,
        )

        # Average price as a dashed line
        plt.axhline(
            self.agent.average_price,
            color="teal",
            linestyle="--",
            label="Average Price",
            linewidth=2,
        )

        # Plot stock levels
        plt.plot(
            self.environment.stock_history,
            label="Stock Level",
            color="darkgreen",
            linewidth=2,
        )

        # Plot units purchased as bold points
        plt.scatter(
            range(len(self.agent.buy_history)),
            self.agent.buy_history,
            label="Units Purchased",
            color="orange",
            edgecolor="black",
            s=100,  # Size of the points
            zorder=3,  # Ensure points appear above lines
        )

        # Titles and labels
        plt.title("Smartphone Price, Stock Levels, and Purchases Over Time")
        plt.ylabel("Value")
        plt.xlabel("Time Step")
        plt.legend()

        plt.tight_layout()
        plt.show()




# Environment Class
class SmartphoneEnvironment:
    price_delta = [10, -20, 5, -15, 0, 25, -30, -20, -5, 0, 5, 20]
    noise_sd = 5

    def __init__(self):
        self.time = 0
        self.stock = 20
        self.price = 300.00
        self.stock_history = [self.stock]
        self.price_history = [self.price]

    def initial_percept(self):
        return {"price": self.price, "stock": self.stock}

    def do_action(self, action):
        daily_sales = select_from_dist({3: 0.2, 5: 0.3, 7: 0.3, 10: 0.2})
        bought = action.get("buy", 0)
        self.stock = max(0, self.stock + bought - daily_sales)
        self.time += 1
        self.price += (
            self.price_delta[self.time % len(self.price_delta)]
            + random.gauss(0, self.noise_sd)
        )
        self.stock_history.append(self.stock)
        self.price_history.append(self.price)
        return {"price": self.price, "stock": self.stock}


# Controller Classes
class PriceMonitoringController:
    def __init__(self, agent, discount_threshold=0.2):
        self.agent = agent
        self.discount_threshold = discount_threshold

    def monitor(self, percept):
        current_price = percept["price"]
        if current_price < (1 - self.discount_threshold) * self.agent.average_price:
            return True
        return False


class InventoryMonitoringController:
    def __init__(self, threshold=15):
        self.threshold = threshold

    def monitor(self, percept):
        return percept["stock"] < self.threshold


class OrderingController:
    def __init__(self, price_controller, inventory_controller):
        self.price_controller = price_controller
        self.inventory_controller = inventory_controller

    def order(self, percept):
        current_price = percept["price"]
        # Check for discount and low stock condition
        if self.price_controller.monitor(percept) and not self.inventory_controller.monitor(percept):
            discount_ratio = (self.price_controller.agent.average_price - current_price) / self.price_controller.agent.average_price
            tobuy = int(15 * (1 + discount_ratio))  # Buy more based on discount size
            print(f"Discount detected! Discount ratio: {discount_ratio:.2f}. Ordering {tobuy} units.")
            return tobuy
        elif self.inventory_controller.monitor(percept):
            print("Low stock detected. Ordering 10 units.\n\n")
            return 10
        print("No action taken. No significant discount or stock issue.\n\n")
        return 0


# Agent Class
class SmartphoneAgent:
    def __init__(self):
        self.average_price = 300  # Initial average price
        self.buy_history = []  # Tracks buying decisions
        self.total_spent = 0  # Tracks total expenditure

        # Controllers
        self.price_controller = PriceMonitoringController(self)
        self.inventory_controller = InventoryMonitoringController()
        self.ordering_controller = OrderingController(self.price_controller, self.inventory_controller)

    def select_action(self, percept):
        current_price = percept["price"]
        self.average_price += (current_price - self.average_price) * 0.1

        # Determine monitoring results
        price_discount = self.price_controller.monitor(percept)
        low_stock = self.inventory_controller.monitor(percept)

        # Use the ordering controller to decide how many units to buy
        tobuy = self.ordering_controller.order(percept)

        # Track expenditure and decisions
        self.total_spent += tobuy * current_price
        self.buy_history.append(tobuy)

        # Print detailed decision log
        print(f"Time: {len(self.buy_history) - 1}")
        print(f"Price: {current_price:.0f}, Stock: {percept['stock']}")
        print(
            f"Price Monitoring: {'Discount detected' if price_discount else 'No discount'} "
            f"(Price: {current_price}, Average: {self.average_price:.0f})"
        )
        print(
            f"Inventory Monitoring: {'Low stock detected' if low_stock else 'Sufficient stock'} "
            f"(Stock: {percept['stock']})"
        )
        print(f"Action: Order {tobuy} units\n")

        return {"buy": tobuy}


# Simulation Class
class Simulation:
    def __init__(self, agent, environment):
        self.agent = agent
        self.environment = environment
        self.percept = self.environment.initial_percept()

    def run(self, steps):
        for step in range(steps):
            action = self.agent.select_action(self.percept)
            self.percept = self.environment.do_action(action)


# Main Simulation
if __name__ == "__main__":
    environment = SmartphoneEnvironment()
    agent = SmartphoneAgent()
    simulation = Simulation(agent, environment)
    simulation.run(50)
    plotter = PlotHistory(agent, environment)
    plotter.plot_history()