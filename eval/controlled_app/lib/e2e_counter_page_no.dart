import 'dart:async';
import 'package:flutter/material.dart';

import 'models.dart';
import 'screens/auth_screens.dart';
import 'screens/main_screens.dart';
import 'screens/restaurant_and_cart_screens.dart';
import 'screens/checkout_and_status_screens.dart';

enum AppScreen {
  login,
  signup,
  requestLocation,
  home,
  restaurant,
  cart,
  paymentMethod,
  checkout,
  orderStatus,
}

class E2eCounterPageNo extends StatefulWidget {
  const E2eCounterPageNo({super.key});

  @override
  State<E2eCounterPageNo> createState() => _E2eCounterPageNoState();
}

class _E2eCounterPageNoState extends State<E2eCounterPageNo> {
  // Navigation State
  AppScreen _currentScreen = AppScreen.requestLocation;

  // Location State
  bool _locationPermissionGranted = false;

  // Accessibility State (simulating large font scale for audits)
  double _fontScale = 1.0;

  // Selected Restaurant
  Restaurant? _selectedRestaurant;

  // Cart State (itemId -> quantity)
  final Map<String, int> _cart = {};

  // Payment State
  String _selectedPaymentMethod = 'card'; // card, cash

  // Checkout Controllers & State
  final _addressController = TextEditingController();
  final _cardNumberController = TextEditingController();
  final _cvvController = TextEditingController();
  String? _checkoutError;
  int _placeOrderTapCount = 0;
  DateTime? _lastPlaceOrderTapTime;
  bool _multiTapWarningVisible = false;

  // Order Status State
  String _orderStatus = 'confirming'; // confirming -> preparing -> on the way
  Timer? _orderStatusTimer;

  @override
  void dispose() {
    _addressController.dispose();
    _cardNumberController.dispose();
    _cvvController.dispose();
    _orderStatusTimer?.cancel();
    super.dispose();
  }

  MenuItem? _findMenuItem(String id) {
    for (var r in dummyRestaurants) {
      for (var item in r.menu) {
        if (item.id == id) return item;
      }
    }
    return null;
  }

  void _placeOrder() {
    final now = DateTime.now();
    setState(() {
      _placeOrderTapCount++;
      if (_lastPlaceOrderTapTime != null) {
        final diff = now.difference(_lastPlaceOrderTapTime!).inMilliseconds;
        if (diff < 400) {
          _multiTapWarningVisible = true;
        }
      }
      _lastPlaceOrderTapTime = now;
    });

    final address = _addressController.text;
    final card = _cardNumberController.text.replaceAll(' ', '');
    final cvv = _cvvController.text;

    if (address.isEmpty) {
      setState(() => _checkoutError = 'Address is required');
      return;
    }

    if (_selectedPaymentMethod == 'card') {
      if (card.length < 15 || card.length > 19) {
        setState(() => _checkoutError = 'Invalid card number');
        return;
      }
      if (cvv.length < 3) {
        setState(() => _checkoutError = 'Invalid CVV');
        return;
      }
    }

    // Success
    setState(() {
      _checkoutError = null;
      _orderStatus = 'confirming';
      _currentScreen = AppScreen.orderStatus;
    });
    _startOrderStatusTimer();
  }

  void _startOrderStatusTimer() {
    _orderStatusTimer?.cancel();
    _orderStatusTimer = Timer.periodic(const Duration(seconds: 3), (timer) {
      if (!mounted) return;
      setState(() {
        if (_orderStatus == 'confirming') {
          _orderStatus = 'preparing';
        } else if (_orderStatus == 'preparing') {
          _orderStatus = 'on the way';
          _orderStatusTimer?.cancel();
        }
      });
    });
  }

  void _advanceOrderStatus() {
    setState(() {
      if (_orderStatus == 'confirming') {
        _orderStatus = 'preparing';
      } else if (_orderStatus == 'preparing') {
        _orderStatus = 'on the way';
      }
    });
  }

  void _resetApp() {
    setState(() {
      _cart.clear();
      _addressController.clear();
      _cardNumberController.clear();
      _cvvController.clear();
      _placeOrderTapCount = 0;
      _multiTapWarningVisible = false;
      _checkoutError = null;
      _currentScreen = AppScreen.login;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            // Accessibility controls & Global Actions always visible for visual test harnesses
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              color: Colors.grey.shade100,
              child: Row(
                children: [
                  const Icon(
                    Icons.accessibility_new,
                    size: 20,
                    color: Colors.blueGrey,
                  ),
                  const SizedBox(width: 8),
                  const Text('Font Scale:', style: TextStyle(fontSize: 12)),
                  Expanded(
                    child: Slider(
                      // key: const Key('font_scale_slider'),
                      value: _fontScale,
                      min: 0.8,
                      max: 2.5,
                      onChanged: (val) => setState(() => _fontScale = val),
                    ),
                  ),
                  TextButton.icon(
                    // key: const Key('global_reset_button'),
                    icon: const Icon(Icons.refresh, size: 16),
                    label: const Text('Reset', style: TextStyle(fontSize: 12)),
                    onPressed: _resetApp,
                  ),
                ],
              ),
            ),
            Expanded(child: _buildCurrentScreen()),
          ],
        ),
      ),
    );
  }

  Widget _buildCurrentScreen() {
    switch (_currentScreen) {
      case AppScreen.login:
        return LoginScreen(
          fontScale: _fontScale,
          onLoginSuccess: (email, password) {
            setState(() => _currentScreen = AppScreen.requestLocation);
          },
          onGoToSignup: () {
            setState(() => _currentScreen = AppScreen.signup);
          },
        );
      case AppScreen.signup:
        return SignupScreen(
          fontScale: _fontScale,
          onSignupSuccess: (name, phone) {
            setState(() => _currentScreen = AppScreen.requestLocation);
          },
          onGoToLogin: () {
            setState(() => _currentScreen = AppScreen.login);
          },
        );
      case AppScreen.requestLocation:
        return RequestLocationScreen(
          fontScale: _fontScale,
          onPermissionResponse: (granted) {
            setState(() {
              _locationPermissionGranted = granted;
              _currentScreen = AppScreen.home;
            });
          },
        );
      case AppScreen.home:
        return HomeScreen(
          fontScale: _fontScale,
          locationPermissionGranted: _locationPermissionGranted,
          onToggleLocationPermission: () {
            setState(
              () => _locationPermissionGranted = !_locationPermissionGranted,
            );
          },
          onSelectRestaurant: (rest) {
            setState(() {
              _selectedRestaurant = rest;
              _currentScreen = AppScreen.restaurant;
            });
          },
        );
      case AppScreen.restaurant:
        return RestaurantScreen(
          fontScale: _fontScale,
          restaurant: _selectedRestaurant!,
          cart: _cart,
          onAddItem: (itemId) {
            setState(() {
              _cart[itemId] = (_cart[itemId] ?? 0) + 1;
            });
          },
          onBack: () {
            setState(() => _currentScreen = AppScreen.home);
          },
          onViewCart: () {
            setState(() => _currentScreen = AppScreen.cart);
          },
        );
      case AppScreen.cart:
        return CartScreen(
          fontScale: _fontScale,
          cart: _cart,
          onBack: () {
            setState(() => _currentScreen = AppScreen.restaurant);
          },
          onIncreaseQty: (itemId) {
            setState(() {
              _cart[itemId] = (_cart[itemId] ?? 0) + 1;
            });
          },
          onDecreaseQty: (itemId) {
            setState(() {
              if (_cart[itemId]! > 1) {
                _cart[itemId] = _cart[itemId]! - 1;
              } else {
                _cart.remove(itemId);
              }
            });
          },
          onDoubleItems: () {
            setState(() {
              _cart.updateAll((key, val) => val * 2);
            });
          },
          onProceedToCheckout: () {
            setState(() => _currentScreen = AppScreen.paymentMethod);
          },
          findMenuItem: _findMenuItem,
        );
      case AppScreen.paymentMethod:
        return PaymentMethodScreen(
          fontScale: _fontScale,
          selectedPaymentMethod: _selectedPaymentMethod,
          onPaymentMethodChanged: (val) {
            setState(() => _selectedPaymentMethod = val);
          },
          onBack: () {
            setState(() => _currentScreen = AppScreen.cart);
          },
          onContinue: () {
            setState(() => _currentScreen = AppScreen.checkout);
          },
        );
      case AppScreen.checkout:
        return CheckoutScreen(
          fontScale: _fontScale,
          selectedPaymentMethod: _selectedPaymentMethod,
          addressController: _addressController,
          cardNumberController: _cardNumberController,
          cvvController: _cvvController,
          checkoutError: _checkoutError,
          placeOrderTapCount: _placeOrderTapCount,
          multiTapWarningVisible: _multiTapWarningVisible,
          onPlaceOrder: _placeOrder,
          onBack: () {
            setState(() => _currentScreen = AppScreen.paymentMethod);
          },
        );
      case AppScreen.orderStatus:
        return OrderStatusScreen(
          fontScale: _fontScale,
          orderStatus: _orderStatus,
          onAdvanceStatus: _advanceOrderStatus,
          onReset: _resetApp,
        );
    }
  }
}
