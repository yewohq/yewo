# Release-candidate applications

Real third-party apps for exercising Yewo against something harder than
`eval/controlled_app`. The controlled app is a small Flutter project with a flat
widget tree; these are shipping applications with the ambiguity, density and
custom UI that actually break perception and grounding.

```
android/   21 APKs    — 20 downloaded release builds, 1 built from source
ios/        3 .app    — Simulator bundles, all built from source
../journeys/yewo.yaml — settings and 12 executable workflows
```

Two apps ship as ABI-split downloads, so VLC and Element X each appear twice,
once per architecture.

Four of the Android packages are two defect pairs rather than release builds.
Trainer App appears as `1645-buggy` / `1645-fixed` and `1640-buggy` /
`1640-fixed`, taken from DroidFixBench, where they are `andror2-1645` and
`andror2-1640`. Each is a known defect beside the developer's fix for it: same
app, same journey, one difference. Every other app here is a working release,
which measures whether Yewo navigates and whether it keeps quiet. The fixed
half of a pair is the only thing in this directory that can tell you a finding
was not invented.

`1645` crashes when the Angle box is checked in the Add Exercise dialog, and
its fix changes one method over two lines, so the two builds differ by 92
bytes. `1640` renders the licence on the Info screen incomplete, but its fix
landed 1,783 version codes later and adds the `INTERNET` permission, so that
pair differs by much more than the defect and is the weaker measurement.

## Materialize the corpus

The deployable packages are intentionally not stored in Git. Their exact
versions, official sources, sizes, architectures, licences, and hashes are in
[`manifest.yaml`](manifest.yaml). Prepare the Android corpus with:

```bash
uv run yewo eval setup --platform android
```

The command keeps an already valid package, downloads a missing package from
its pinned upstream release, and refuses any bytes that do not match the
manifest. Use `--verify-only` to perform no network access.

Locally built artifacts have no download URL. The command verifies them when
present and otherwise reports the build instructions below. This distinction
prevents Yewo from silently redistributing third-party builds whose publication
conditions and embedded configuration have not been audited.

One `yewo.yaml` holds the settings and every workflow. A workflow owns an app
and its environment; the journeys inside it are only the tests. Run it from
the repository root as shown below.

Each journey names the screens and controls the app actually shows, taken from
a first run on a cleared install rather than from its documentation. Where an
app asks for a permission or walks through a setup flow, the journey includes
it, because that is what the agent will meet.

One apk does not run on an arm64 emulator: `wonderous-2.2.7-x86_64.apk`
installs and then crashes, because its Flutter engine is x86_64 only. Pick the
`arm64` file for VLC and Element X and both install and run. Wonderous iOS runs
on the Simulator, so the cross-platform pair below needs an arm64 Android
rebuild, not a new app.

Element X and Bluesky have no journeys, because both need a real account.
The three iOS bundles have no journeys either, because no iOS journey can run
on this machine yet — see "Driving the iOS bundles" below.

Run one from the repository root with:

```bash
uv run yewo test eval/journeys/yewo.yaml --workflow tasks
uv run yewo test eval/journeys/yewo.yaml
```

| Workflow | App | What it is there for |
| --- | --- | --- |
| `fossify_calendar` | Fossify Calendar | The control. Plain native UI, no permission, no account |
| `fossify_notes` | Fossify Notes | Text entry that must survive leaving the editor |
| `tasks` | Tasks | Setup choice first, then a task editor of many fields |
| `markor` | Markor | Storage permission that routes through the Settings app |
| `fossify_gallery` | Fossify Gallery | A media permission dialog from the permission controller |
| `organic_maps` | Organic Maps | Location permission, a 70 MB download, then a canvas with no tree |
| `wikipedia` | Wikipedia | Search, article, and saving, against live content |
| `antennapod` | AntennaPod | Search and subscribe, so the result depends on the network |
| `mihon` | Mihon | A Compose setup flow with named themes |
| `ankidroid` | AnkiDroid | Deck and card creation, the densest UI here |
| `bitwarden` | Bitwarden | Stops at the sign-in screen, having no account |
| `vlc` | VLC | Both permission kinds in one onboarding, then a video player |

## Why these apps

Screen count is not what makes an app hard for a UI agent. What matters is
**what the app renders into**, because that decides whether Appium's page
source gives you anything usable. These apps were chosen to cover every
rendering backend rather than to pile up more of the same kind:

| Backend | What the accessibility tree gives you | Apps here |
| --- | --- | --- |
| Android views (XML) | Real `resource-id`s, class names, bounds — easy mode | Tasks, AntennaPod, Fossify x3, Wikipedia, AnkiDroid, VLC |
| Jetpack Compose | A semantics tree, no resource-ids, synthetic nodes | Mihon, Element X, Bitwarden |
| React Native | Bridged nodes, inconsistent labelling | Bluesky |
| Flutter | Flat semantics, frequently unlabelled | Wonderous |
| Custom canvas | Nothing. Pure vision and OCR | Organic Maps |
| UIKit (+ some SwiftUI) | Rich tree, real identifiers | NetNewsWire |
| SwiftUI | Large runs of `XCUIElementTypeOther`, no labels — worse than Compose | Ice Cubes |

**Wonderous ships on both platforms.** That pair is the most useful thing in
here: when a journey passes on Android and fails on iOS, the app is not the
variable. Only the iOS half runs today. The Android apk is x86_64 only, so the
pair needs an arm64 rebuild before you can use it this way.

## Android

`ABI` is what the apk actually carries. `universal` means all four of
`arm64-v8a`, `armeabi-v7a`, `x86`, `x86_64`.

| File | Package | Size | ABI | Account | License |
| --- | --- | --- | --- | --- | --- |
| `org.tasks_151202.apk` | `org.tasks` | 27M | universal | no | GPL-3.0 |
| `net.gsantner.markor_163.apk` | `net.gsantner.markor` | 11M | no native code | no | see repo |
| `de.danoeh.antennapod_3120395.apk` | `de.danoeh.antennapod` | 17M | universal | no | GPL-3.0 |
| `org.fossify.calendar_23.apk` | `org.fossify.calendar` | 7.8M | universal | no | GPL-3.0 |
| `org.fossify.notes_13.apk` | `org.fossify.notes` | 9.1M | universal | no | GPL-3.0 |
| `org.fossify.gallery_29.apk` | `org.fossify.gallery` | 37M | universal | no | GPL-3.0 |
| `org.wikipedia_50610.apk` | `org.wikipedia` | 79M | universal | no | Apache-2.0 |
| `com.ichi2.anki_22502300.apk` | `com.ichi2.anki` | 107M | universal | no | GPL-3.0 |
| `app.organicmaps_26092932.apk` | `app.organicmaps` | 61M | universal | no | see repo |
| `org.videolan.vlc_13070106_arm64.apk` | `org.videolan.vlc` | 46M | `arm64-v8a` | no | GPL (VideoLAN) |
| `org.videolan.vlc_13070108.apk` | `org.videolan.vlc` | 47M | `x86_64` | no | GPL (VideoLAN) |
| `mihon-0.21.2-foss.apk` | `app.mihon.foss` | 64M | universal | no | Apache-2.0 |
| `com.x8bit.bitwarden-2026.10.0.apk` | `com.x8bit.bitwarden` | 89M | universal + mips | **yes** | GPL-3.0 |
| `element-x-26.10.0-arm64.apk` | `io.element.android.x` | 136M | `arm64-v8a` | **yes** | AGPL-3.0 |
| `element-x-26.10.0-x86_64.apk` | `io.element.android.x` | 147M | `x86_64` | **yes** | AGPL-3.0 |
| `bluesky-1.135.0.apk` | `xyz.blueskyweb.app` | 259M | universal | **yes** | MIT |
| `wonderous-2.2.7-x86_64.apk` | `com.gskinner.flutter.wonders` | 65M | `x86_64` engine | no | MIT |
| `trainer-app-1645-buggy.apk` | `com.german_software_engineers.trainerapp` | 2.4M | no native code | no | GPL-3.0 |
| `trainer-app-1645-fixed.apk` | `com.german_software_engineers.trainerapp` | 2.4M | no native code | no | GPL-3.0 |
| `trainer-app-1640-buggy.apk` | `com.german_software_engineers.trainerapp` | 2.4M | no native code | no | GPL-3.0 |
| `trainer-app-1640-fixed.apk` | `com.german_software_engineers.trainerapp` | 2.7M | no native code | no | GPL-3.0 |

### What each one is and what to test

**Tasks** — A to-do app, and the best starting point. No account at all, but it
has due-date and recurrence pickers, priorities, tags, saved filters,
multi-select and a deep settings tree. Date and time pickers are where
mark-based grounding usually falls over first.

**Markor** — Plain-text and Markdown notes over a file browser, with a
formatting toolbar. Small, fast, fully offline. Heavy text entry.

**AntennaPod** — Podcast client. Adds network dependence, search-and-subscribe,
a drag-to-reorder queue, playback controls and a sleep timer. Good for gestures
and for journeys that span several screens.

**Fossify Calendar / Notes / Gallery** — The maintained successors to Simple
Mobile Tools. Plain, unambiguous native UI, so they make good controls: if a
journey fails here, it is probably the agent and not the app. Calendar is the
cheapest way to test event creation with date/time entry.

**Wikipedia** — A genuinely polished production app. Search with live
suggestions, article navigation, tabs, reading lists, themes and language
settings. The "does this hold up on something real" test. No login needed for
most journeys.

**AnkiDroid** — Where to go when you want the agent to struggle. Deeply nested
preferences, modal dialogs, a card editor, decks, and plenty of non-obvious
affordances. Heavy enough that emulator speed shows up as flake.

**Organic Maps** — Offline maps. The map itself is a custom-drawn canvas, so the
accessibility tree tells you nothing and grounding is vision-only. Search,
routing and bookmarks are the testable flows. Needs a region download first.

**VLC** — Media player. Library scanning, equalizer, subtitle and audio track
pickers, and swipe-gesture player controls. Push some media onto the device
first. Two ABI builds are here; use the `arm64` one on an Apple Silicon
emulator.

VLC's onboarding is the most obstacle-dense first run in this corpus, and the
obstacles come from four different owners:

1. The app's own wizard (`OnboardingActivity`).
2. **The Settings app**, for all-files access — `com.android.settings`, not the
   permission controller. Same route Markor takes.
3. **The permission controller**, for notifications — the ordinary dialog.
4. `com.android.systemui`, which shows a "Viewing full screen" cling the first
   time the player goes immersive, plus VLC's own "Video player tips" overlay.

That makes it the sharpest test of whether permission and overlay handling keys
on the right package. A guard that only knows `permissioncontroller` clears
step 3 and fails step 2.

The All files access page is Compose and exposes **no `clickable="true"` node at
all** — the switch is visible to the eye and absent from the tree, so grounding
there has to be vision-only.

The `vlc` workflow expects video files on the device. Push some first:

```bash
adb push <clips> /sdcard/Movies/
adb shell am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE \
  -d file:///sdcard/Movies
```

**Mihon** — Compose, no account. A library grid, source/extension management,
and a custom gesture reader with paged and webtoon modes. The reader is close to
unsolvable from the accessibility tree alone, which makes it a good vision test.
Content requires installing a source extension.

**Bitwarden** — The best target for Yewo's secret-masking and redaction
features specifically: real password fields, show/hide toggles, a generator,
autofill. This is the `-fdroid` variant, so it has no Google Play Services
dependency and runs on a plain AOSP emulator image. Requires an account.

**Element X** — A Matrix client, fully rewritten in Compose. Rooms, timelines,
media, settings. Pairs with `ios/` if you later build Element X iOS. Needs a
Matrix account (matrix.org is free). Two ABI builds are here; use the `arm64`
one on an Apple Silicon emulator.

**Bluesky** — The React Native data point, and a real production social app.
Feeds, composing with media, profiles, notifications, settings. Universal APK
(all ABIs), which is why it is the largest file here. Needs an account.

**Wonderous** — A Flutter showcase app: parallax scrolling, custom-painted
illustrations, timeline scrubbers, photo galleries and an embedded WebView.
Visually the most complex thing here and almost entirely custom-drawn, so the
semantics tree is thin. Fully offline, no account. Built from source — see
"Rebuilding Wonderous" below.

This apk **does not run on an arm64 emulator**. It installs, then dies on the
first activity with `Could not find 'libflutter.so'`. Use the iOS bundle, or
rebuild for arm64. The iOS half runs today and is the better target anyway:
same app, richer screens, no rebuild needed.

## iOS

All three are **debug Simulator builds**. They will not install on a physical
device, and they are unsigned.

| Bundle | Executable | Bundle ID | Size | Account | License |
| --- | --- | --- | --- | --- | --- |
| `NetNewsWire.app` | `NetNewsWire` | `com.ranchero.NetNewsWire.iOS-DEBUG` | 33M | no | MIT |
| `Ice Cubes.app` | `Ice Cubes` | `com.ronin.test.IceCubesApp` | 222M | **yes** | AGPL-3.0 |
| `Wonderous.app` | `Runner` | `com.gskinner.flutter.wonders` | 264M | no | MIT |

**NetNewsWire** — A mature RSS reader, mostly UIKit with some SwiftUI. Works
with no account: add a feed URL and you have a real content hierarchy to
navigate. Its Feedly, Inoreader and Mercury integrations are inert because this
build was compiled with empty API secrets, which does not affect local feeds.

**Ice Cubes** — A Mastodon client and the best *modern SwiftUI* target available.
Almost entirely SwiftUI, which means XCUITest reports large runs of
`XCUIElementTypeOther` with no label or identifier — the iOS analogue of the
Compose problem, and worse. Needs a Mastodon account.

**Wonderous** — Same app as the Android build above, so use the pair to separate
platform bugs from app bugs. Universal binary (x86_64 + arm64), so it runs on
Intel and Apple Silicon simulators alike. Minimum iOS 13.0. Verified on an
iPhone 17 Pro Max simulator running iOS 26.3: it opens on a three-page
onboarding carousel that advances by horizontal swipe. The Android half does
not run on arm64, so this bundle is the only working Wonderous today.

## Driving the iOS bundles

All three bundles install and launch. Yewo cannot drive any of them on this
machine yet, because the Appium **XCUITest driver is not installed** — only
`uiautomator2` is:

```bash
appium driver list --installed     # uiautomator2@7.5.2 only
appium driver install xcuitest     # what is missing
```

`idb` is not installed either, and `simctl` injects no taps or swipes. So
`simctl` alone can install, launch, screenshot and uninstall, but it cannot
reach a second screen. That is why the iOS bundles have no journeys: the
journeys here are written from the running app, and nothing can currently walk
an iOS app past its first screen.

What is verified today, by `simctl install` and `simctl launch`:

All three were verified on an iPhone 17 Pro Max simulator running iOS 26.3:

| Bundle | Installs | Launches | First screen |
| --- | --- | --- | --- |
| `Wonderous.app` | yes | yes | Three-page onboarding carousel, advanced by swipe |
| `NetNewsWire.app` | yes | yes | Stays running |
| `Ice Cubes.app` | yes | yes | **Add Account**: an Instance URL field over a list of suggested Mastodon servers, plus an iOS notification alert |

Ice Cubes fetches that server list over the network, so its first screen is not
offline and its content will differ between runs.

## Resetting state between runs

Android, per package, keeps the app installed:

```bash
adb shell pm clear org.tasks
```

iOS has no `pm clear` equivalent:

```bash
xcrun simctl uninstall <udid> com.ranchero.NetNewsWire.iOS-DEBUG   # one app
xcrun simctl erase <udid>                                          # whole device
```

Prefer `pm clear` over reinstalling — several of these apps persist state in
ways that quietly change a second run's starting screen.

## Gotchas

- **`Ice Cubes.app` has a space** in both the bundle and the executable name,
  and the executable is `Ice Cubes`, not `IceCubesApp`. Quote the path.
- **Wonderous's iOS executable is `Runner`**, Flutter's default. The bundle was
  renamed to `Wonderous.app` so it is distinguishable from
  `eval/controlled_app`, which also builds a `Runner.app`.
- **Mihon's package is `app.mihon.foss`**, not `app.mihon` — this is the FOSS
  flavour.
- **VLC on F-Droid is ABI-split** across four version codes sharing one version
  name. `13070108` is x86_64 and `13070106` is arm64-v8a. Both are here. The
  version code, not the version name, tells you which architecture you have.
- **Check the ABI before you call an app broken.** An Apple Silicon emulator is
  `arm64-v8a`, and an x86_64 apk fails there with
  `INSTALL_FAILED_NO_MATCHING_ABIS`. That is the wrong file, not a broken app.
  VLC and Element X each ship an arm64 build, and both install and run.
- **`aapt dump badging` can overstate the ABIs of a Flutter apk.** It reports
  every `lib/<abi>/` directory present, and a plugin that ships all four makes
  a single-ABI build look universal. Wonderous reports `arm64-v8a armeabi-v7a
  x86_64`, but only `lib/x86_64/` holds `libflutter.so` and `libapp.so`; the
  arm64 directory holds one 7KB plugin. Check for `libflutter.so` by ABI:

  ```bash
  unzip -l app.apk | grep libflutter.so
  ```
- **XCUITest page-source snapshots are slow** — often 1-3 seconds each versus
  tens of milliseconds for uiautomator. Per-step timing budgets tuned on Android
  will not hold on iOS.
- **No iOS equivalent of F-Droid exists.** iOS targets must be built from
  source for the Simulator; `.ipa` distribution requires signing. Expect one
  undocumented bootstrap step per iOS app (NetNewsWire needed
  `buildscripts/updateSecrets.sh`; Ice Cubes needed an `IceCubesApp.xcconfig`).

## Provenance

The 16 downloaded APKs came from f-droid.org/repo with version codes resolved
through the F-Droid API, or from the project's own GitHub releases (Bitwarden,
Element X, Mihon, Bluesky). Filenames pin exact versions. Bitwarden's published
SHA-256 was verified against the download.

Everything in `ios/` plus `wonderous-2.2.7-x86_64.apk` was built from source on
2026-10-16 with Xcode 26.3 and Flutter 3.44.8. The source checkouts were
deleted afterwards; the notes below are what it takes to reproduce them.

### Rebuilding Wonderous

`gskinnerTeam/flutter-wonderous-app` does not publish APKs and its checked-in
toolchain is too old to build as-is. iOS needs nothing beyond
`flutter build ios --simulator` (debug only — `--release` is rejected for
simulators). Android needs four changes:

1. `android/settings.gradle.kts` — Kotlin plugin `2.1.0` to `2.2.20`.
   Otherwise `google_maps_flutter_android` crashes the Kotlin compiler.
2. `android/app/build.gradle.kts` — `compileSdk = 37` instead of
   `flutter.compileSdkVersion`, required by `androidx.glance`.
3. `android/settings.gradle.kts` and `gradle-wrapper.properties` — AGP `8.9.1`
   to `8.13.0`, Gradle `8.13` to `8.14.3`. AGP 8.9.1 looks for
   `platforms/android-37` and cannot read a minor-versioned `android-37.0` SDK
   directory.
4. `android/build.gradle.kts` — pin glance and raise the JVM target, **above**
   the existing `subprojects { project.evaluationDependsOn(":app") }` block,
   because that call forces evaluation and finalises both settings:

   ```kotlin
   subprojects {
       configurations.configureEach {
           resolutionStrategy.force("androidx.glance:glance-appwidget:1.1.1")
       }
       afterEvaluate {
           extensions.findByName("android")?.let { ext ->
               (ext as com.android.build.gradle.BaseExtension).compileOptions {
                   sourceCompatibility = JavaVersion.VERSION_17
                   targetCompatibility = JavaVersion.VERSION_17
               }
           }
           tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile>()
               .configureEach {
                   compilerOptions.jvmTarget.set(
                       org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
               }
       }
   }
   ```

   The root cause of (4) is that `home_widget` 0.9.1 declares
   `androidx.glance:glance-appwidget:1.+`, an open range that now resolves to
   `1.3.0-alpha02` and demands AGP 9.1.0, and compiles its own Kotlin at JVM
   target 1.8 while inlining dependencies built higher. JVM 17 rather than 11:
   `google_maps_flutter_android` uses Java 17 pattern matching. Deleting
   `home_widget` from `pubspec.yaml` is the simpler alternative, since the
   home-screen widget is irrelevant to UI testing.

### Rebuilding the iOS apps

- **NetNewsWire** — run `./buildscripts/updateSecrets.sh` first to gyb-generate
  the gitignored `Modules/Secrets/Sources/Secrets/SecretKey.swift`. Skip
  `setup.sh`; it is interactive and only needed for device signing. Then
  `xcodebuild -project NetNewsWire.xcodeproj -scheme NetNewsWire-iOS
  -destination 'platform=iOS Simulator,name=<device>' CODE_SIGNING_ALLOWED=NO build`.
- **Ice Cubes** — built from **tag 2.1.3**, not `main`. HEAD uses SwiftUI
  `ToolbarContent.axisBehavior` and `.verticalPreferred`, which do not exist in
  Xcode 26.3's SDK. Copy `IceCubesApp.xcconfig.template` to
  `IceCubesApp.xcconfig`; `DEVELOPMENT_TEAM` can be empty for Simulator builds,
  and `BUNDLE_ID_PREFIX` here was set to `com.ronin.test`, which is where that
  bundle ID comes from.

## Not included, but worth knowing

- [AndroidWorld](https://github.com/google-research/android_world) — Google's
  agent benchmark: 116 tasks over 20 Android apps, several of which are in this
  directory, with **programmatic reward functions** per task. The most useful
  thing available for checking a verifier against ground truth rather than
  against the agent's own self-report.
- MobileAgentBench and LlamaTouch annotate intermediate states, which is closer
  to what a step-level verifier wants than final-state-only rewards.
- There is **no comparable public iOS benchmark**. Ground truth on iOS has to be
  authored by hand.
