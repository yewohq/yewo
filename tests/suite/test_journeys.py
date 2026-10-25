"""Reading a journey file, and the values it is allowed to carry."""

import unittest



class JourneyEnvTests(unittest.TestCase):
    """A journey declares its values and says which are secret."""

    def _journey(self, data):
        from yewo.testing.suite import _journey

        return _journey(data, {}, 1)

    def test_placeholders_are_replaced_in_the_journey_text(self):
        journey = self._journey(
            {
                "env": {"email": "qa@example.com"},
                "journey": "Type {{email}} in the Email field",
            }
        )

        self.assertEqual(journey.description, "Type qa@example.com in the Email field")

    def test_only_named_secrets_are_collected(self):
        journey = self._journey(
            {
                "env": {"email": "qa@example.com", "password": "hunter2"},
                "secrets": ["password"],
                "journey": "Sign in",
            }
        )

        self.assertEqual(journey.secrets, {"password": "hunter2"})

    def test_a_secret_naming_no_declared_value_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            self._journey({"env": {"email": "a@b.com"}, "secrets": ["password"]})

        self.assertIn("password", str(caught.exception))

    def test_values_can_come_from_the_environment(self):
        import os

        from yewo.testing.journey_env import resolve

        os.environ["YEWO_ENV_TEST"] = "from-the-shell"
        try:
            _, secrets, _ = resolve(
                {"env": {"password": "${YEWO_ENV_TEST}"}, "secrets": ["password"]}
            )
        finally:
            del os.environ["YEWO_ENV_TEST"]

        self.assertEqual(secrets, {"password": "from-the-shell"})

    def test_security_rules_see_the_same_values(self):
        journey = self._journey(
            {
                "env": {"card": "4111222233334444"},
                "journey": "Pay",
                "security_rules": ["{{card}} must never be shown in full"],
            }
        )

        self.assertEqual(
            journey.security_rules, ["4111222233334444 must never be shown in full"]
        )

    def test_a_journey_without_an_env_block_still_loads(self):
        journey = self._journey({"journey": "Sign in", "security_rules": ["No leaks"]})

        self.assertEqual(journey.secrets, {})
        self.assertEqual(journey.description, "Sign in")


class InlineSecretTests(unittest.TestCase):
    """A value can say it is secret where it is declared."""

    def _resolve(self, data):
        from yewo.testing.journey_env import resolve

        return resolve(data)

    def test_a_value_can_mark_itself_secret(self):
        values, secrets, _ = self._resolve(
            {
                "env": {
                    "email": "qa@example.com",
                    "password": {"value": "hunter2", "secret": True},
                }
            }
        )

        self.assertEqual(values["password"], "hunter2")
        self.assertEqual(secrets, {"password": "hunter2"})

    def test_a_secret_can_choose_its_model_display(self):
        from yewo.testing.journey_env import secret_displays

        displays = secret_displays(
            {
                "env": {
                    "card": {
                        "value": "4111222233334444",
                        "secret": True,
                        "redaction": "last4",
                    }
                }
            }
        )

        self.assertEqual(displays, {"card": "last4"})

    def test_an_unknown_secret_display_is_rejected(self):
        from yewo.testing.journey_env import secret_displays

        with self.assertRaises(ValueError):
            secret_displays(
                {
                    "env": {
                        "password": {
                            "value": "hunter2",
                            "secret": True,
                            "redaction": "sometimes",
                        }
                    }
                }
            )

    def test_an_inline_value_defaults_to_ordinary(self):
        _, secrets, _ = self._resolve({"env": {"note": {"value": "visible"}}})

        self.assertEqual(secrets, {})

    def test_inline_and_listed_secrets_combine(self):
        _, secrets, _ = self._resolve(
            {
                "env": {
                    "password": {"value": "hunter2", "secret": True},
                    "cvv": "321",
                },
                "secrets": ["cvv"],
            }
        )

        self.assertEqual(secrets, {"cvv": "321", "password": "hunter2"})

    def test_a_mapping_without_a_value_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            self._resolve({"env": {"password": {"secret": True}}})

        self.assertIn("value", str(caught.exception))

    def test_a_misspelled_key_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            self._resolve({"env": {"password": {"value": "x", "secrets": True}}})

        self.assertIn("secrets", str(caught.exception))


class ValueVerificationPolicyTests(unittest.TestCase):
    def test_an_app_can_declare_its_own_mask_and_normalization(self):
        from yewo.testing.journey_env import verification_policies

        policies = verification_policies(
            {
                "env": {
                    "account_reference": {
                        "value": "AB-1234",
                        "verify_as": "masked",
                        "normalizers": ["hyphens", "case"],
                        "visible_suffix": 2,
                        "masked_length": 6,
                    }
                }
            }
        )

        policy = policies["account_reference"]
        self.assertEqual(policy.mode, "masked")
        self.assertEqual(policy.normalizers, ("hyphens", "case"))
        self.assertEqual(policy.visible_suffix, 2)
        self.assertEqual(policy.masked_length, 6)

    def test_comparison_options_require_a_compatible_mode(self):
        from yewo.testing.journey_env import verification_policies

        with self.assertRaises(ValueError) as caught:
            verification_policies(
                {
                    "env": {
                        "reference": {
                            "value": "A1",
                            "verify_as": "exact",
                            "visible_suffix": 1,
                        }
                    }
                }
            )

        self.assertIn("verify_as is exact", str(caught.exception))


class MissingEnvTests(unittest.TestCase):
    """A blank password is a configuration error, not a test of blank passwords."""

    def test_an_unset_variable_fails_the_journey(self):
        import os

        from yewo.testing.journey_env import resolve

        os.environ.pop("YEWO_DEFINITELY_UNSET", None)

        with self.assertRaises(ValueError) as caught:
            resolve({"env": {"password": "${YEWO_DEFINITELY_UNSET}"}})

        self.assertIn("YEWO_DEFINITELY_UNSET", str(caught.exception))
