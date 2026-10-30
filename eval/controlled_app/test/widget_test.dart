import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:testing_app/main.dart';

void main() {
  testWidgets('completes the food delivery flow', (WidgetTester tester) async {
    await tester.pumpWidget(const MyApp());
    await tester.tap(find.text('Reset'));
    await tester.pumpAndSettle();

    expect(find.text('Yewo Fast Food Delivery'), findsOneWidget);
    await tester.enterText(find.byType(TextField).at(0), 'test@example.com');
    await tester.enterText(find.byType(TextField).at(1), 'password123');
    await tester.tap(find.text('Login'));
    await tester.pumpAndSettle();

    expect(find.text('Allow Location Access'), findsOneWidget);
    await tester.tap(find.text('Deny & Alphabetical Sorting'));
    await tester.pumpAndSettle();

    expect(find.textContaining('Location Denied'), findsOneWidget);
    await tester.tap(find.text('Menu').first);
    await tester.pumpAndSettle();

    await tester.tap(find.text('Add').first);
    await tester.pump();
    await tester.tap(find.text('Cart (1)'));
    await tester.pumpAndSettle();

    await tester.tap(find.text('Proceed to Checkout'));
    await tester.pumpAndSettle();

    expect(find.text('Select Payment'), findsOneWidget);
    await tester.tap(find.text('Continue'));
    await tester.pumpAndSettle();

    await tester.enterText(find.byType(TextField).at(0), '123 Test Street');
    await tester.enterText(find.byType(TextField).at(1), '4111222233334444');
    await tester.enterText(find.byType(TextField).at(2), '123');
    await tester.ensureVisible(find.text('Place Order'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Place Order'));
    await tester.pumpAndSettle();

    expect(find.textContaining('CONFIRMING'), findsOneWidget);
  });
}
