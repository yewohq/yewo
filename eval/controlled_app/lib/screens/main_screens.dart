import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import '../models.dart';

class RequestLocationScreen extends StatelessWidget {
  final double fontScale;
  final Function(bool granted) onPermissionResponse;

  static const _platform = MethodChannel('app/native_permissions');

  const RequestLocationScreen({
    super.key,
    required this.fontScale,
    required this.onPermissionResponse,
  });

  Future<void> _requestLocationPermission() async {
    try {
      final bool? granted = await _platform.invokeMethod<bool>('requestLocationPermission');
      onPermissionResponse(granted ?? true);
    } catch (_) {
      onPermissionResponse(true);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(32),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const Icon(Icons.location_on, size: 80, color: Colors.deepOrange),
          const SizedBox(height: 24),
          Text(
            'Allow Location Access',
            style: TextStyle(
              fontSize: 22 * fontScale,
              fontWeight: FontWeight.bold,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 16),
          const Text(
            'We use your location to find and sort the closest restaurants near you. Please grant permission to continue.',
            textAlign: TextAlign.center,
            style: TextStyle(color: Colors.grey),
          ),
          const SizedBox(height: 48),
          ElevatedButton(
            // key: const Key('grant_location_permission'),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.green,
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(vertical: 16),
            ),
            onPressed: _requestLocationPermission,
            child: const Text('Grant Location Permission'),
          ),
          const SizedBox(height: 16),
          OutlinedButton(
            // key: const Key('deny_location_permission'),
            style: OutlinedButton.styleFrom(
              side: const BorderSide(color: Colors.red),
              foregroundColor: Colors.red,
              padding: const EdgeInsets.symmetric(vertical: 16),
            ),
            onPressed: () => onPermissionResponse(false),
            child: const Text('Deny & Alphabetical Sorting'),
          ),
        ],
      ),
    );
  }
}

class HomeScreen extends StatelessWidget {
  final double fontScale;
  final bool locationPermissionGranted;
  final VoidCallback onToggleLocationPermission;
  final Function(Restaurant rest) onSelectRestaurant;

  const HomeScreen({
    super.key,
    required this.fontScale,
    required this.locationPermissionGranted,
    required this.onToggleLocationPermission,
    required this.onSelectRestaurant,
  });

  List<Restaurant> _getSortedRestaurants() {
    final list = List<Restaurant>.from(dummyRestaurants);
    if (locationPermissionGranted) {
      list.sort((a, b) => a.distance.compareTo(b.distance));
    } else {
      list.sort((a, b) => a.name.compareTo(b.name));
    }
    return list;
  }

  @override
  Widget build(BuildContext context) {
    final sortedList = _getSortedRestaurants();

    return Column(
      children: [
        // Location status bar
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          color: locationPermissionGranted
              ? Colors.green.shade50
              : Colors.red.shade50,
          child: Row(
            // key: const Key('location_status'),
            children: [
              Icon(
                locationPermissionGranted ? Icons.gps_fixed : Icons.gps_off,
                color: locationPermissionGranted ? Colors.green : Colors.red,
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  locationPermissionGranted
                      ? 'Location: Active (Nearest First)'
                      : 'Location Denied: Sorting Alphabetically',
                  style: TextStyle(
                    color: locationPermissionGranted
                        ? Colors.green.shade800
                        : Colors.red.shade800,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              ElevatedButton(
                // key: const Key('toggle_location_permission'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: locationPermissionGranted
                      ? Colors.red
                      : Colors.green,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 8,
                  ),
                ),
                onPressed: onToggleLocationPermission,
                child: Text(
                  locationPermissionGranted ? 'Deny' : 'Grant',
                  style: const TextStyle(fontSize: 11),
                ),
              ),
            ],
          ),
        ),
        Expanded(
          child: ListView.builder(
            itemCount: sortedList.length,
            itemBuilder: (context, index) {
              final rest = sortedList[index];
              return Card(
                margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                child: ListTile(
                  leading: CircleAvatar(
                    backgroundColor: Colors.deepOrange.shade100,
                    child: Text(
                      rest.name[0],
                      style: const TextStyle(color: Colors.deepOrange),
                    ),
                  ),
                  title: Text(
                    rest.name,
                    style: const TextStyle(fontWeight: FontWeight.bold),
                  ),
                  subtitle: Text(
                    '${rest.category} • ${rest.distance.toStringAsFixed(1)} km',
                  ),
                  trailing: ElevatedButton(
                    // key: Key('restaurant_card_${index}'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.deepOrange,
                      foregroundColor: Colors.white,
                    ),
                    onPressed: () => onSelectRestaurant(rest),
                    child: const Text('Menu'),
                  ),
                ),
              );
            },
          ),
        ),
      ],
    );
  }
}
