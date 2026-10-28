import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:testing_app/main.dart';

void main() {
  testWidgets('Full Food Delivery Flow Smoke Test', (WidgetTester tester) async {
    // 1. Build our app and trigger a frame.
    await tester.pumpWidget(const MyApp());

    // Verify Login Screen elements
    expect(find.byKey(const Key('login_email')), findsOneWidget);
    expect(find.byKey(const Key('login_password')), findsOneWidget);
    expect(find.byKey(const Key('login_button')), findsOneWidget);

    // Enter login credentials
    await tester.enterText(find.byKey(const Key('login_email')), 'test@example.com');
    await tester.enterText(find.byKey(const Key('login_password')), 'password123');
    await tester.tap(find.byKey(const Key('login_button')));
    await tester.pumpAndSettle();

    // 2. Should transition to Request Location Screen
    expect(find.byKey(const Key('grant_location_permission')), findsOneWidget);
    expect(find.byKey(const Key('deny_location_permission')), findsOneWidget);

    // Grant Location Permission
    await tester.tap(find.byKey(const Key('grant_location_permission')));
    await tester.pumpAndSettle();

    // 3. Should transition to Home Screen (Restaurant list)
    expect(find.byKey(const Key('location_status')), findsOneWidget);
    expect(find.textContaining('Location: Active'), findsOneWidget);
    expect(find.byKey(const Key('restaurant_card_0')), findsOneWidget); // Closest restaurant card

    // Tap first restaurant (Sushi Zen or Burger Palace)
    await tester.tap(find.byKey(const Key('restaurant_card_0')));
    await tester.pumpAndSettle();

    // 4. Should transition to Restaurant Menu Screen
    expect(find.byKey(const Key('view_cart_button')), findsOneWidget);

    // Add first menu item to cart
    final addButtonFinder = find.byKey(const Key('add_to_cart_item_st_1'));
    expect(addButtonFinder, findsOneWidget);
    await tester.tap(addButtonFinder);
    await tester.pump();

    // View Cart
    await tester.tap(find.byKey(const Key('view_cart_button')));
    await tester.pumpAndSettle();

    // 5. Should transition to Cart Screen
    expect(find.byKey(const Key('proceed_to_checkout')), findsOneWidget);
    expect(find.byKey(const Key('double_items_button')), findsOneWidget);

    // Proceed to Payment Method Selection
    await tester.tap(find.byKey(const Key('proceed_to_checkout')));
    await tester.pumpAndSettle();

    // 6. Should transition to Payment Method Screen
    expect(find.byKey(const Key('select_payment_card')), findsOneWidget);
    expect(find.byKey(const Key('select_payment_cash')), findsOneWidget);
    expect(find.byKey(const Key('continue_to_payment_details')), findsOneWidget);

    // Choose Credit Card (default) and Continue
    await tester.tap(find.byKey(const Key('continue_to_payment_details')));
    await tester.pumpAndSettle();

    // 7. Should transition to Checkout Screen
    expect(find.byKey(const Key('checkout_address')), findsOneWidget);
    expect(find.byKey(const Key('checkout_card_number')), findsOneWidget);
    expect(find.byKey(const Key('checkout_cvv')), findsOneWidget);
    expect(find.byKey(const Key('place_order_button')), findsOneWidget);

    // Fill checkout details
    await tester.enterText(find.byKey(const Key('checkout_address')), '123 Test Street, QA City');
    await tester.enterText(find.byKey(const Key('checkout_card_number')), '4111222233334444');
    await tester.enterText(find.byKey(const Key('checkout_cvv')), '123');

    // Scroll to Place Order button
    await tester.ensureVisible(find.byKey(const Key('place_order_button')));
    await tester.pumpAndSettle();

    // Place Order
    await tester.tap(find.byKey(const Key('place_order_button')));
    await tester.pumpAndSettle();

    // 8. Should transition to Order Status Screen
    expect(find.byKey(const Key('order_status_label')), findsOneWidget);
    expect(find.textContaining('CONFIRMING'), findsOneWidget);
  });
}
