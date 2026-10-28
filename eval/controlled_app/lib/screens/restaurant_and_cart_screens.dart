import 'package:flutter/material.dart';
import '../models.dart';

class RestaurantScreen extends StatelessWidget {
  final double fontScale;
  final Restaurant restaurant;
  final Map<String, int> cart;
  final Function(String itemId) onAddItem;
  final VoidCallback onBack;
  final VoidCallback onViewCart;

  const RestaurantScreen({
    super.key,
    required this.fontScale,
    required this.restaurant,
    required this.cart,
    required this.onAddItem,
    required this.onBack,
    required this.onViewCart,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(restaurant.name),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: onBack,
        ),
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  restaurant.category,
                  style: const TextStyle(fontSize: 16, color: Colors.grey),
                ),
                Text(
                  '${restaurant.distance} km away',
                  style: const TextStyle(fontSize: 16, color: Colors.grey),
                ),
              ],
            ),
          ),
          const Divider(),
          Expanded(
            child: ListView.builder(
              itemCount: restaurant.menu.length,
              itemBuilder: (context, index) {
                final item = restaurant.menu[index];
                final cartQty = cart[item.id] ?? 0;
                return Card(
                  margin: const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 8,
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: Row(
                      children: [
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                item.name,
                                style: const TextStyle(
                                  fontWeight: FontWeight.bold,
                                  fontSize: 16,
                                ),
                              ),
                              const SizedBox(height: 4),
                              Text(
                                item.description,
                                style: const TextStyle(
                                  color: Colors.grey,
                                  fontSize: 12,
                                ),
                              ),
                              const SizedBox(height: 8),

                              // Price Box with fixed dimensions to trigger text clipping during scale audits!
                              Container(
                                width: 70,
                                height: 24,
                                alignment: Alignment.centerLeft,
                                decoration: BoxDecoration(
                                  color: Colors.orange.shade50,
                                  borderRadius: BorderRadius.circular(4),
                                ),
                                child: Text(
                                  '\$${item.price.toStringAsFixed(2)}',
                                  // key: Key('price_label_${item.id}'),
                                  style: TextStyle(
                                    fontSize: 14 * fontScale,
                                    fontWeight: FontWeight.bold,
                                    color: Colors.deepOrange,
                                  ),
                                  maxLines: 1,
                                  overflow: TextOverflow
                                      .clip, // Deliberate clipping visual bug!
                                ),
                              ),
                            ],
                          ),
                        ),
                        Column(
                          children: [
                            if (cartQty > 0)
                              Container(
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 8,
                                  vertical: 4,
                                ),
                                decoration: BoxDecoration(
                                  color: Colors.deepOrange,
                                  borderRadius: BorderRadius.circular(12),
                                ),
                                child: Text(
                                  '$cartQty in Cart',
                                  style: const TextStyle(
                                    color: Colors.white,
                                    fontSize: 11,
                                  ),
                                ),
                              ),
                            const SizedBox(height: 8),
                            ElevatedButton(
                              // key: Key('add_to_cart_item_${item.id}'),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Colors.deepOrange,
                                foregroundColor: Colors.white,
                              ),
                              onPressed: () => onAddItem(item.id),
                              child: const Text('Add'),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        // key: const Key('view_cart_button'),
        backgroundColor: Colors.deepOrange,
        foregroundColor: Colors.white,
        onPressed: onViewCart,
        label: Text('Cart (${cart.values.fold(0, (a, b) => a + b)})'),
        icon: const Icon(Icons.shopping_cart),
      ),
    );
  }
}

class CartScreen extends StatelessWidget {
  final double fontScale;
  final Map<String, int> cart;
  final VoidCallback onBack;
  final Function(String itemId) onIncreaseQty;
  final Function(String itemId) onDecreaseQty;
  final VoidCallback onDoubleItems;
  final VoidCallback onProceedToCheckout;
  final MenuItem? Function(String itemId) findMenuItem;

  const CartScreen({
    super.key,
    required this.fontScale,
    required this.cart,
    required this.onBack,
    required this.onIncreaseQty,
    required this.onDecreaseQty,
    required this.onDoubleItems,
    required this.onProceedToCheckout,
    required this.findMenuItem,
  });

  double _getCartSubtotal() {
    double sub = 0;
    cart.forEach((itemId, qty) {
      final item = findMenuItem(itemId);
      if (item != null) {
        sub += item.price * qty;
      }
    });
    return sub;
  }

  @override
  Widget build(BuildContext context) {
    final cartItems = cart.entries.where((e) => e.value > 0).toList();
    final subtotal = _getCartSubtotal();

    return Scaffold(
      appBar: AppBar(
        title: const Text('My Cart'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: onBack,
        ),
      ),
      body: Column(
        children: [
          // Double Cart items switch (Monkey scenario testing helper)
          Container(
            padding: const EdgeInsets.all(16),
            color: Colors.amber.shade50,
            child: Row(
              children: [
                const Icon(Icons.flash_on, color: Colors.amber),
                const SizedBox(width: 8),
                const Expanded(
                  child: Text(
                    'Monkey Multiplier: Double all item quantities instantly.',
                    style: TextStyle(fontSize: 12),
                  ),
                ),
                ElevatedButton(
                  // key: const Key('double_items_button'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.amber.shade800,
                    foregroundColor: Colors.white,
                  ),
                  onPressed: onDoubleItems,
                  child: const Text('x2 Items'),
                ),
              ],
            ),
          ),
          Expanded(
            child: cartItems.isEmpty
                ? const Center(
                    child: Text(
                      'Your cart is empty',
                      style: TextStyle(fontSize: 16, color: Colors.grey),
                    ),
                  )
                : ListView.builder(
                    itemCount: cartItems.length,
                    itemBuilder: (context, index) {
                      final entry = cartItems[index];
                      final item = findMenuItem(entry.key);
                      if (item == null) return const SizedBox.shrink();

                      return Card(
                        margin: const EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 8,
                        ),
                        child: ListTile(
                          title: Text(
                            item.name,
                            style: const TextStyle(fontWeight: FontWeight.bold),
                          ),
                          subtitle: Text(
                            '\$${item.price.toStringAsFixed(2)} x ${entry.value}',
                          ),
                          trailing: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              IconButton(
                                // key: Key('decrease_qty_${item.id}'),
                                icon: const Icon(
                                  Icons.remove_circle_outline,
                                  color: Colors.red,
                                ),
                                onPressed: () => onDecreaseQty(item.id),
                              ),
                              Text(
                                '${entry.value}',
                                style: const TextStyle(
                                  fontSize: 16,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              IconButton(
                                // key: Key('increase_qty_${item.id}'),
                                icon: const Icon(
                                  Icons.add_circle_outline,
                                  color: Colors.green,
                                ),
                                onPressed: () => onIncreaseQty(item.id),
                              ),
                            ],
                          ),
                        ),
                      );
                    },
                  ),
          ),
          Container(
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              color: Colors.white,
              boxShadow: [
                BoxShadow(
                  color: Colors.grey.shade200,
                  blurRadius: 10,
                  spreadRadius: 5,
                ),
              ],
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const Text(
                      'Subtotal:',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    Text(
                      '\$${subtotal.toStringAsFixed(2)}',
                      // key: const Key('cart_subtotal'),
                      style: const TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                        color: Colors.deepOrange,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                if (subtotal == 0)
                  Container(
                    // key: const Key('empty_cart_error'),
                    padding: const EdgeInsets.all(8),
                    color: Colors.red.shade50,
                    margin: const EdgeInsets.only(bottom: 8),
                    child: const Text(
                      'Cannot checkout with an empty cart!',
                      style: TextStyle(color: Colors.red),
                      textAlign: TextAlign.center,
                    ),
                  ),
                ElevatedButton(
                  // key: const Key('proceed_to_checkout'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.deepOrange,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 16),
                  ),
                  onPressed: onProceedToCheckout,
                  child: Text(
                    'Proceed to Checkout',
                    style: TextStyle(fontSize: 16 * fontScale),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
