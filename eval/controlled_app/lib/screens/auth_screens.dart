import 'package:flutter/material.dart';

class LoginScreen extends StatefulWidget {
  final double fontScale;
  final Function(String email, String password) onLoginSuccess;
  final VoidCallback onGoToSignup;

  const LoginScreen({
    super.key,
    required this.fontScale,
    required this.onLoginSuccess,
    required this.onGoToSignup,
  });

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  String? _loginError;

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  void _login() {
    final email = _emailController.text;
    final password = _passwordController.text;

    if (!email.contains('@') || !email.contains('.')) {
      setState(() => _loginError = 'Invalid email address');
      return;
    }
    if (password.length < 6) {
      setState(() => _loginError = 'Password must be at least 6 characters');
      return;
    }

    setState(() => _loginError = null);
    widget.onLoginSuccess(email, password);
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.fastfood, size: 64, color: Colors.deepOrange),
          const SizedBox(height: 16),
          Text(
            'Yewo Fast Food Delivery',
            style: TextStyle(
              fontSize: 24 * widget.fontScale,
              fontWeight: FontWeight.bold,
              color: Colors.deepOrange,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 8),
          const Text('Sign in to continue', textAlign: TextAlign.center),
          const SizedBox(height: 32),
          TextField(
            // key: const Key('login_email'),
            controller: _emailController,
            keyboardType: TextInputType.emailAddress,
            decoration: const InputDecoration(
              labelText: 'Email Address',
              border: OutlineInputBorder(),
              prefixIcon: Icon(Icons.email),
            ),
          ),
          const SizedBox(height: 16),
          TextField(
            // key: const Key('login_password'),
            controller: _passwordController,
            obscureText: true,
            decoration: const InputDecoration(
              labelText: 'Password',
              border: OutlineInputBorder(),
              prefixIcon: Icon(Icons.lock),
            ),
          ),
          if (_loginError != null) ...[
            const SizedBox(height: 16),
            Text(_loginError!, style: const TextStyle(color: Colors.red)),
          ],
          const SizedBox(height: 24),
          ElevatedButton(
            // key: const Key('login_button'),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.deepOrange,
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(vertical: 16),
            ),
            onPressed: _login,
            child: Text(
              'Login',
              style: TextStyle(fontSize: 16 * widget.fontScale),
            ),
          ),
          const SizedBox(height: 16),
          TextButton(
            // key: const Key('go_to_signup'),
            onPressed: widget.onGoToSignup,
            child: const Text("Don't have an account? Sign Up"),
          ),
        ],
      ),
    );
  }
}

class SignupScreen extends StatefulWidget {
  final double fontScale;
  final Function(String name, String phone) onSignupSuccess;
  final VoidCallback onGoToLogin;

  const SignupScreen({
    super.key,
    required this.fontScale,
    required this.onSignupSuccess,
    required this.onGoToLogin,
  });

  @override
  State<SignupScreen> createState() => _SignupScreenState();
}

class _SignupScreenState extends State<SignupScreen> {
  final _nameController = TextEditingController();
  final _phoneController = TextEditingController();
  final _otpController = TextEditingController();
  bool _otpSent = false;
  String? _signupError;

  @override
  void dispose() {
    _nameController.dispose();
    _phoneController.dispose();
    _otpController.dispose();
    super.dispose();
  }

  void _sendOtp() {
    if (_phoneController.text.length < 10) {
      setState(() => _signupError = 'Invalid phone number');
      return;
    }
    setState(() {
      _otpSent = true;
      _signupError = null;
    });
  }

  void _signup() {
    final name = _nameController.text;
    final phone = _phoneController.text;
    final otp = _otpController.text;

    if (name.isEmpty) {
      setState(() => _signupError = 'Name is required');
      return;
    }
    if (phone.length < 10) {
      setState(() => _signupError = 'Phone is too short');
      return;
    }
    if (otp != '1234') {
      setState(() => _signupError = 'Invalid OTP (use 1234)');
      return;
    }

    setState(() => _signupError = null);
    widget.onSignupSuccess(name, phone);
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const SizedBox(height: 40),
          const Icon(Icons.person_add, size: 64, color: Colors.deepOrange),
          const SizedBox(height: 16),
          Text(
            'Create Account',
            style: TextStyle(
              fontSize: 24 * widget.fontScale,
              fontWeight: FontWeight.bold,
              color: Colors.deepOrange,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 32),
          TextField(
            // key: const Key('signup_name'),
            controller: _nameController,
            decoration: const InputDecoration(
              labelText: 'Full Name',
              border: OutlineInputBorder(),
              prefixIcon: Icon(Icons.person),
            ),
          ),
          const SizedBox(height: 16),
          TextField(
            // key: const Key('signup_phone'),
            controller: _phoneController,
            keyboardType: TextInputType.phone,
            decoration: const InputDecoration(
              labelText: 'Phone Number',
              border: OutlineInputBorder(),
              prefixIcon: Icon(Icons.phone),
            ),
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: TextField(
                  // key: const Key('signup_otp'),
                  controller: _otpController,
                  keyboardType: TextInputType.number,
                  decoration: const InputDecoration(
                    labelText: 'OTP Verification Code',
                    border: OutlineInputBorder(),
                    prefixIcon: Icon(Icons.security),
                  ),
                ),
              ),
              const SizedBox(width: 8),
              ElevatedButton(
                // key: const Key('send_otp_button'),
                onPressed: _sendOtp,
                child: const Text('Send OTP'),
              ),
            ],
          ),
          if (_otpSent) ...[
            const SizedBox(height: 8),
            const Text(
              'Simulated OTP code sent! Use 1234',
              style: TextStyle(color: Colors.green, fontSize: 12),
            ),
          ],
          if (_signupError != null) ...[
            const SizedBox(height: 16),
            Text(_signupError!, style: const TextStyle(color: Colors.red)),
          ],
          const SizedBox(height: 32),
          ElevatedButton(
            // key: const Key('signup_button'),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.deepOrange,
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(vertical: 16),
            ),
            onPressed: _signup,
            child: Text(
              'Register & Continue',
              style: TextStyle(fontSize: 16 * widget.fontScale),
            ),
          ),
          const SizedBox(height: 16),
          TextButton(
            // key: const Key('go_to_login'),
            onPressed: widget.onGoToLogin,
            child: const Text("Already have an account? Sign In"),
          ),
        ],
      ),
    );
  }
}
