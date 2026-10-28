class MenuItem {
  final String id;
  final String name;
  final String description;
  final double price;

  MenuItem({
    required this.id,
    required this.name,
    required this.description,
    required this.price,
  });
}

class Restaurant {
  final String id;
  final String name;
  final String category;
  final double distance; // in km
  final List<MenuItem> menu;

  Restaurant({
    required this.id,
    required this.name,
    required this.category,
    required this.distance,
    required this.menu,
  });
}

final List<Restaurant> dummyRestaurants = [
  Restaurant(
    id: '1',
    name: 'Burger Palace',
    category: 'Fast Food',
    distance: 1.2,
    menu: [
      MenuItem(id: 'bp_1', name: 'Classic Burger', description: 'Juicy beef patty with cheese', price: 9.99),
      MenuItem(id: 'bp_2', name: 'French Fries', description: 'Crispy golden fries', price: 3.49),
      MenuItem(id: 'bp_3', name: 'Vanilla Shake', description: 'Creamy vanilla milkshake', price: 4.99),
    ],
  ),
  Restaurant(
    id: '2',
    name: 'Pizza Heaven',
    category: 'Italian',
    distance: 2.5,
    menu: [
      MenuItem(id: 'ph_1', name: 'Pepperoni Pizza', description: 'Loads of pepperoni and cheese', price: 14.99),
      MenuItem(id: 'ph_2', name: 'Garlic Bread', description: 'Buttery garlic bread sticks', price: 5.99),
      MenuItem(id: 'ph_3', name: 'Tiramisu', description: 'Classic Italian coffee dessert', price: 6.99),
    ],
  ),
  Restaurant(
    id: '3',
    name: 'Sushi Zen',
    category: 'Japanese',
    distance: 0.8,
    menu: [
      MenuItem(id: 'sz_1', name: 'Salmon Roll Combo', description: '8 pieces of salmon maki', price: 18.99),
      MenuItem(id: 'sz_2', name: 'Miso Soup', description: 'Traditional Japanese seaweed soup', price: 3.99),
      MenuItem(id: 'sz_3', name: 'Green Tea Ice Cream', description: 'Sweet matcha ice cream', price: 4.99),
    ],
  ),
  Restaurant(
    id: '4',
    name: 'Taco Fiesta',
    category: 'Mexican',
    distance: 3.1,
    menu: [
      MenuItem(id: 'tf_1', name: 'Beef Tacos Trio', description: 'Three soft tacos with beef', price: 10.99),
      MenuItem(id: 'tf_2', name: 'Chips & Guac', description: 'Freshly made guacamole', price: 5.49),
      MenuItem(id: 'tf_3', name: 'Churros', description: 'Mexican cinnamon pastry', price: 4.49),
    ],
  ),
  Restaurant(
    id: '5',
    name: 'Green Garden',
    category: 'Salads',
    distance: 1.5,
    menu: [
      MenuItem(id: 'gg_1', name: 'Caesar Salad', description: 'Romaine lettuce, parmesan, croutons', price: 11.99),
      MenuItem(id: 'gg_2', name: 'Quinoa Bowl', description: 'Quinoa with roasted veggies', price: 12.99),
      MenuItem(id: 'gg_3', name: 'Fresh Orange Juice', description: '100% freshly squeezed juice', price: 4.99),
    ],
  ),
  Restaurant(
    id: '6',
    name: 'Curry House',
    category: 'Indian',
    distance: 4.2,
    menu: [
      MenuItem(id: 'ch_1', name: 'Chicken Tikka Masala', description: 'Creamy chicken curry', price: 15.99),
      MenuItem(id: 'ch_2', name: 'Garlic Naan', description: 'Clay oven baked flatbread', price: 2.99),
      MenuItem(id: 'ch_3', name: 'Mango Lassi', description: 'Sweet mango yogurt drink', price: 3.99),
    ],
  ),
  Restaurant(
    id: '7',
    name: 'Wok & Roll',
    category: 'Chinese',
    distance: 2.9,
    menu: [
      MenuItem(id: 'wr_1', name: 'Kung Pao Chicken', description: 'Spicy chicken with peanuts', price: 13.99),
      MenuItem(id: 'wr_2', name: 'Veggie Spring Rolls', description: 'Three crispy spring rolls', price: 4.99),
      MenuItem(id: 'wr_3', name: 'Fried Rice', description: 'Wok-tossed rice with egg', price: 8.99),
    ],
  ),
  Restaurant(
    id: '8',
    name: 'Sweet Treats',
    category: 'Desserts',
    distance: 0.5,
    menu: [
      MenuItem(id: 'st_1', name: 'Chocolate Fudge Cake', description: 'Rich chocolate layered cake', price: 7.99),
      MenuItem(id: 'st_2', name: 'Apple Pie', description: 'Warm apple pie with cinnamon', price: 6.49),
      MenuItem(id: 'st_3', name: 'Double Scoop Waffle', description: 'Waffle with ice cream', price: 8.99),
    ],
  ),
  Restaurant(
    id: '9',
    name: 'Pasta Grill',
    category: 'Italian',
    distance: 3.7,
    menu: [
      MenuItem(id: 'pg_1', name: 'Fettuccine Alfredo', description: 'Pasta in rich parmesan cream', price: 14.99),
      MenuItem(id: 'pg_2', name: 'Bruschetta', description: 'Grilled bread topped with tomatoes', price: 6.99),
      MenuItem(id: 'pg_3', name: 'Cannoli Trio', description: 'Sweet ricotta pastry tubes', price: 5.99),
    ],
  ),
  Restaurant(
    id: '10',
    name: 'The Daily Grind',
    category: 'Coffee',
    distance: 1.9,
    menu: [
      MenuItem(id: 'dg_1', name: 'Caffe Latte', description: 'Espresso with steamed milk', price: 4.49),
      MenuItem(id: 'dg_2', name: 'Butter Croissant', description: 'Flaky baked pastry', price: 3.29),
      MenuItem(id: 'dg_3', name: 'Blueberry Muffin', description: 'Freshly baked muffin', price: 3.79),
    ],
  ),
];
