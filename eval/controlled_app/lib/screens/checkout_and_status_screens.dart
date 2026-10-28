import 'package:flutter/material.dart';

class PaymentMethodScreen extends StatelessWidget {
  final double fontScale;
  final String selectedPaymentMethod;
  final Function(String val) onPaymentMethodChanged;
  final VoidCallback onBack;
  final VoidCallback onContinue;

  const PaymentMethodScreen({
    super.key,
    required this.fontScale,
    required this.selectedPaymentMethod,
    required this.onPaymentMethodChanged,
    required this.onBack,
    required this.onContinue,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Select Payment'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: onBack,
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text(
              'Choose your preferred payment method:',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 24),
            RadioListTile<String>(
              // key: const Key('select_payment_card'),
              title: const Text('Credit / Debit Card'),
              subtitle: const Text(
                'Pay securely with Visa, Mastercard, or Amex',
              ),
              value: 'card',
              groupValue: selectedPaymentMethod,
              onChanged: (val) => onPaymentMethodChanged(val!),
            ),
            const Divider(),
            RadioListTile<String>(
              // key: const Key('select_payment_cash'),
              title: const Text('Cash on Delivery (COD)'),
              subtitle: const Text('Pay with cash when your food arrives'),
              value: 'cash',
              groupValue: selectedPaymentMethod,
              onChanged: (val) => onPaymentMethodChanged(val!),
            ),
            const Spacer(),
            ElevatedButton(
              // key: const Key('continue_to_payment_details'),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.deepOrange,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 16),
              ),
              onPressed: onContinue,
              child: Text(
                'Continue',
                style: TextStyle(fontSize: 16 * fontScale),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class CheckoutScreen extends StatefulWidget {
  final double fontScale;
  final String selectedPaymentMethod;
  final TextEditingController addressController;
  final TextEditingController cardNumberController;
  final TextEditingController cvvController;
  final String? checkoutError;
  final int placeOrderTapCount;
  final bool multiTapWarningVisible;
  final VoidCallback onPlaceOrder;
  final VoidCallback onBack;

  const CheckoutScreen({
    super.key,
    required this.fontScale,
    required this.selectedPaymentMethod,
    required this.addressController,
    required this.cardNumberController,
    required this.cvvController,
    required this.checkoutError,
    required this.placeOrderTapCount,
    required this.multiTapWarningVisible,
    required this.onPlaceOrder,
    required this.onBack,
  });

  @override
  State<CheckoutScreen> createState() => _CheckoutScreenState();
}

class _CheckoutScreenState extends State<CheckoutScreen> {
  String _getMaskedCardNumberDisplay(String input) {
    final clean = input.replaceAll(RegExp(r'\D'), '');
    if (clean.isEmpty) return '•••• •••• •••• ••••';

    final formatted = StringBuffer();
    for (int i = 0; i < clean.length; i++) {
      if (i > 0 && i % 4 == 0) {
        formatted.write(' ');
      }
      if (i < clean.length - 4) {
        formatted.write('•');
      } else {
        formatted.write(clean[i]);
      }
    }

    while (formatted.length < 19) {
      if (formatted.length == 4 ||
          formatted.length == 9 ||
          formatted.length == 14) {
        formatted.write(' ');
      } else {
        formatted.write('•');
      }
    }

    return formatted.toString();
  }

  @override
  Widget build(BuildContext context) {
    final isCard = widget.selectedPaymentMethod == 'card';

    return Scaffold(
      appBar: AppBar(
        title: const Text('Checkout Details'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: widget.onBack,
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text(
              'Delivery Address',
              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
            ),
            const SizedBox(height: 8),
            TextField(
              // key: const Key('checkout_address'),
              controller: widget.addressController,
              maxLines: 3,
              maxLength: 600,
              decoration: const InputDecoration(
                hintText: 'Enter your full delivery address here...',
                border: OutlineInputBorder(),
              ),
              onChanged: (val) => setState(() {}),
            ),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  'Address Length Indicator:',
                  style: TextStyle(fontSize: 12, color: Colors.grey),
                ),
                Text(
                  '${widget.addressController.text.length} chars',
                  //key: const Key('address_length'),
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                    color: widget.addressController.text.length > 500
                        ? Colors.red
                        : Colors.green,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 24),

            // Payment Details (visible/active if card selected)
            if (isCard) ...[
              const Text(
                'Credit Card Details',
                style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
              ),
              const SizedBox(height: 16),

              // Custom interactive Masked Card visual preview (prevents plain text visibility)
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [Colors.deepOrange, Colors.orange.shade800],
                  ),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'SECURE PAYMENT CARD',
                      style: TextStyle(
                        color: Colors.white70,
                        fontSize: 10,
                        letterSpacing: 1.5,
                      ),
                    ),
                    const SizedBox(height: 20),
                    Text(
                      _getMaskedCardNumberDisplay(
                        widget.cardNumberController.text,
                      ),
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 20,
                        letterSpacing: 2.0,
                        fontFamily: 'Courier',
                      ),
                    ),
                    const SizedBox(height: 20),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        const Text(
                          'YEWO TESTER',
                          style: TextStyle(color: Colors.white70, fontSize: 12),
                        ),
                        Text(
                          'CVV: ${"•" * widget.cvvController.text.length}',
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 12,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),

              TextField(
                // key: const Key('checkout_card_number'),
                controller: widget.cardNumberController,
                keyboardType: TextInputType.number,
                obscureText: true, // Obscured for security compliance
                decoration: const InputDecoration(
                  labelText: 'Card Number',
                  border: OutlineInputBorder(),
                  prefixIcon: Icon(Icons.credit_card),
                ),
                onChanged: (val) => setState(() {}),
              ),
              const SizedBox(height: 16),
              TextField(
                // key: const Key('checkout_cvv'),
                controller: widget.cvvController,
                keyboardType: TextInputType.number,
                obscureText: true,
                decoration: const InputDecoration(
                  labelText: 'CVV',
                  border: OutlineInputBorder(),
                  prefixIcon: Icon(Icons.security),
                ),
                onChanged: (val) => setState(() {}),
              ),
            ],

            if (widget.checkoutError != null) ...[
              const SizedBox(height: 16),
              Text(
                widget.checkoutError!,
                key: const Key('card_validation_error'),
                style: const TextStyle(
                  color: Colors.red,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],

            if (widget.multiTapWarningVisible) ...[
              const SizedBox(height: 16),
              Container(
                // key: const Key('multi_tap_warning'),
                padding: const EdgeInsets.all(12),
                color: Colors.red.shade100,
                child: Text(
                  'Multi-tap detected! Total place order button taps: ${widget.placeOrderTapCount}',
                  style: const TextStyle(
                    color: Colors.red,
                    fontWeight: FontWeight.bold,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
            ],

            const SizedBox(height: 32),
            ElevatedButton(
              // key: const Key('place_order_button'),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.green,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 16),
              ),
              onPressed: widget.onPlaceOrder,
              child: Text(
                'Place Order',
                style: TextStyle(
                  fontSize: 18 * widget.fontScale,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Taps Registered: ${widget.placeOrderTapCount}',
              //key: const Key('orders_placed_count'),
              style: const TextStyle(fontSize: 11, color: Colors.grey),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}

class OrderStatusScreen extends StatelessWidget {
  final double fontScale;
  final String orderStatus;
  final VoidCallback onAdvanceStatus;
  final VoidCallback onReset;

  const OrderStatusScreen({
    super.key,
    required this.fontScale,
    required this.orderStatus,
    required this.onAdvanceStatus,
    required this.onReset,
  });

  Widget _buildStatusIndicator() {
    int activeIndex = 0;
    if (orderStatus == 'preparing') activeIndex = 1;
    if (orderStatus == 'on the way') activeIndex = 2;

    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: [
        _buildStatusDot(0, activeIndex, 'Confirmed'),
        _buildStatusLine(activeIndex >= 1),
        _buildStatusDot(1, activeIndex, 'Preparing'),
        _buildStatusLine(activeIndex >= 2),
        _buildStatusDot(2, activeIndex, 'On The Way'),
      ],
    );
  }

  Widget _buildStatusDot(int index, int activeIndex, String label) {
    final isActive = index <= activeIndex;
    return Column(
      children: [
        CircleAvatar(
          radius: 12,
          backgroundColor: isActive ? Colors.deepOrange : Colors.grey.shade300,
          child: Icon(
            Icons.check,
            size: 12,
            color: isActive ? Colors.white : Colors.grey,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          label,
          style: TextStyle(
            fontSize: 9,
            color: isActive ? Colors.black : Colors.grey,
          ),
        ),
      ],
    );
  }

  Widget _buildStatusLine(bool active) {
    return Container(
      width: 40,
      height: 2,
      color: active ? Colors.deepOrange : Colors.grey.shade300,
    );
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(32),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const Icon(Icons.check_circle_outline, size: 80, color: Colors.green),
          const SizedBox(height: 24),
          Text(
            'Order Placed Successfully!',
            style: TextStyle(
              fontSize: 22 * fontScale,
              fontWeight: FontWeight.bold,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 32),
          Container(
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              color: Colors.orange.shade50,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: Colors.deepOrange),
            ),
            child: Column(
              children: [
                const Text(
                  'Current Status:',
                  style: TextStyle(fontSize: 14, color: Colors.grey),
                ),
                const SizedBox(height: 8),
                Text(
                  orderStatus.toUpperCase(),
                  key: const Key('order_status_label'),
                  style: TextStyle(
                    fontSize: 24 * fontScale,
                    fontWeight: FontWeight.bold,
                    color: Colors.deepOrange,
                  ),
                ),
                const SizedBox(height: 16),
                _buildStatusIndicator(),
              ],
            ),
          ),
          const SizedBox(height: 48),
          ElevatedButton(
            key: const Key('advance_status_button'),
            onPressed: onAdvanceStatus,
            child: const Text('Advance Status (Manual)'),
          ),
          const SizedBox(height: 16),
          OutlinedButton(
            onPressed: onReset,
            child: const Text('Order Again (Home)'),
          ),
        ],
      ),
    );
  }
}
