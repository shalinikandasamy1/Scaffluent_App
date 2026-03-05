# Leveraging Claude Code for Rapid Swift App Development: Scaffluent iOS App

## Table of Contents

1. [App Overview](#app-overview)
2. [Strengths of Claude Code for Swift Development](#strengths)
3. [Weaknesses and Limitations](#weaknesses)
4. [Workarounds for Each Weakness](#workarounds)
5. [Recommended Workflow for the Scaffluent App](#recommended-workflow)
6. [Prompt Engineering Tips for Swift/SwiftUI](#prompt-engineering-tips)
7. [Appendix: Scaffluent App Architecture Reference](#appendix)

---

## 1. App Overview <a name="app-overview"></a>

The Scaffluent iOS app is a SwiftUI-based application for conducting real-time safety evaluations of scaffolding structures. It integrates multiple hardware sensors (camera via AVFoundation/ARKit, thermal via FLIR ONE Pro) and performs multi-module analysis (vision, tilt detection, thermal) to produce comprehensive safety assessments.

**Key technical characteristics:**
- **Pure SwiftUI** with Combine for reactive data binding
- **MVVM architecture** with clear separation: Models, ViewModels, Views, Services, Components
- **Hardware integrations**: AVFoundation, ARKit (LiDAR), FLIR ONE Pro SDK
- **~32 Swift source files** across a well-organized directory structure
- **Xcode project** (`.xcodeproj`), no CocoaPods or SPM packages detected

This is a moderately complex app — large enough that AI-assisted development provides significant value, but small enough that Claude Code can maintain full context of the codebase.

---

## 2. Strengths of Claude Code for Swift Development <a name="strengths"></a>

### 2.1 Full-Codebase Context Awareness

Claude Code can read, search, and reason across all ~32 Swift files simultaneously. For the Scaffluent app, this means it can:
- Trace data flow from `LoginViewModel` through `AuthService` to `AppViewModel` state changes
- Understand how `EvaluationViewModel` orchestrates `CameraService`, `ThermalSDKWrapper`, and the `Aggregator`
- Identify where a model change (e.g., adding a field to `Issue`) requires updates across views, view models, and services

This cross-file awareness is something Xcode's autocomplete cannot match.

### 2.2 SwiftUI View Generation

Claude Code excels at generating SwiftUI views. It handles:
- Declarative layout composition (`VStack`, `HStack`, `ZStack`, `List`, `NavigationStack`)
- State management patterns (`@State`, `@Binding`, `@ObservedObject`, `@EnvironmentObject`, `@Observable`)
- Modifier chains and conditional rendering
- Reusable component extraction (the app's `PrimaryButton`, `StatusChipView`, `IssueRowView` patterns)

Real-world evidence: Developers have shipped 20,000+ line macOS apps writing fewer than 1,000 lines by hand, and successfully rewrote 12-year-old Objective-C apps to SwiftUI using Claude Code.

### 2.3 MVVM Boilerplate and Architecture

The Scaffluent app follows a consistent MVVM pattern. Claude Code can:
- Generate new screen/feature pairs (View + ViewModel) following the existing pattern
- Create new model types conforming to `Codable`, `Identifiable`, `Hashable`
- Wire up `@Published` properties and Combine pipelines
- Generate service layer methods with proper async/await signatures

### 2.4 Multi-File Refactoring

Claude Code can perform coordinated changes across many files:
- Rename a type or method across models, view models, views, and services
- Extract shared logic into a new service
- Migrate from `ObservableObject`/`@Published` to `@Observable` macro (iOS 17+)
- Add a new field to a model and update all consumers

Developers report Claude gets projects building again after refactors within 1-2 build-fix cycles.

### 2.5 Test Generation

Claude Code can generate:
- Unit tests for ViewModels and Services (mocking dependencies)
- Snapshot/preview tests for SwiftUI views
- Integration test scaffolding for sensor pipelines

### 2.6 Documentation and Code Explanation

Claude Code can explain complex flows like the evaluation pipeline (`CameraService` -> Vision module -> Aggregator -> `EvaluationResult`) and generate inline documentation following Apple's `///` doc comment conventions.

---

## 3. Weaknesses and Limitations <a name="weaknesses"></a>

### 3.1 Cannot Modify `.pbxproj` Files Safely

**Severity: Critical**

The Xcode project file (`project.pbxproj`) is a complex, proprietary format with UUIDs, build phases, and file references. Claude Code corrupts these files when attempting edits. A single bad edit can require hours of manual recovery.

### 3.2 No Xcode Build System Access on Linux

**Severity: Critical (for this setup)**

The development machine runs Linux. iOS apps require `xcodebuild` on macOS for compilation, linking, code signing, and simulator testing. Claude Code on Linux cannot:
- Compile the Scaffluent app
- Run it in the iOS Simulator
- Execute Xcode Previews
- Run `xcodebuild test`

### 3.3 No Simulator or Device Testing

**Severity: High**

Claude Code cannot launch, interact with, or observe the iOS Simulator. It cannot verify:
- Runtime behavior (navigation flows, animations, state transitions)
- Camera/ARKit/thermal sensor integration on real hardware
- Permission dialog flows
- Memory usage or performance characteristics

### 3.4 Stale Swift/SwiftUI Knowledge

**Severity: Medium**

Swift and SwiftUI see significant API changes yearly. Claude Code may:
- Use deprecated APIs (e.g., `NavigationView` instead of `NavigationStack`)
- Miss newer patterns (e.g., `@Observable` macro vs. `ObservableObject` protocol)
- Over-use `DispatchQueue.main.async` instead of `@MainActor`
- Over-use `GeometryReader` when `containerRelativeFrame()` or `visualEffect()` would be better
- Apply excessive `fontWeight()` modifiers instead of proper font weight parameters

### 3.5 Limited Understanding of Hardware SDK Integrations

**Severity: Medium**

The FLIR ONE Pro SDK (`ThermalSDKWrapper`) and ARKit LiDAR integration involve:
- Vendor-specific APIs with limited public documentation
- Objective-C bridging headers
- Hardware-specific callback patterns
- Device-specific quirks

Claude Code's training data has limited coverage of niche hardware SDKs.

### 3.6 Storyboard/XIB Handling

**Severity: Low (for this app)**

The Scaffluent app is pure SwiftUI, so this is not a concern here. But for mixed UIKit/SwiftUI codebases, Claude Code cannot meaningfully edit Interface Builder files.

### 3.7 Asset Catalog Management

**Severity: Low**

Claude Code cannot add images to `Assets.xcassets` (which requires specific directory structures and `Contents.json` files) or manage app icons, color sets, or other catalog resources effectively.

### 3.8 Code Signing and Provisioning

**Severity: Low (handled by Xcode)**

Claude Code has no awareness of code signing identities, provisioning profiles, or entitlements. These are managed through Xcode and the Apple Developer portal.

---

## 4. Workarounds for Each Weakness <a name="workarounds"></a>

### 4.1 `.pbxproj` Corruption → Folder-Based Project Organization

**Solution**: Use Xcode 16+ folder-based source management.

With folder references, placing a `.swift` file in the correct directory automatically includes it in the build target — no `.pbxproj` edit needed. Configure the Scaffluent project to use folder references for `Models/`, `Screens/`, `Components/`, `Services/`, and `Utils/`.

**Immediate rule**: Add to `CLAUDE.md` in the project root:
```
NEVER modify .pbxproj, .xcodeproj, or .xcworkspace files.
When creating new Swift files, place them in the correct directory.
The developer will add them to the Xcode project manually if needed.
```

### 4.2 No Build System on Linux → Remote Mac Build + XcodeBuildMCP

**Solutions (pick one or combine)**:

1. **SSH to a Mac**: If a Mac is available on the network, use `ssh mac-host 'cd /path/to/project && xcodebuild build -scheme Scaffluent -destination "platform=iOS Simulator,name=iPhone 16"'` to build remotely.

2. **XcodeBuildMCP on Mac**: Install the [XcodeBuildMCP](https://github.com/getsentry/XcodeBuildMCP) MCP server on a Mac. This gives Claude Code direct build/test/preview capabilities through the MCP protocol. Install via `brew install xcodebuildmcp` or `npx`.

3. **CI/CD pipeline**: Push to a branch, let GitHub Actions (with macOS runners) or a local Buildkite agent on a Mac build and report results.

4. **Accept the limitation**: Use Claude Code purely for code authoring on Linux, and build/test on a Mac separately. This is the simplest approach and still captures most of the value.

### 4.3 No Simulator Testing → Screenshot-Driven Feedback Loop

**Solution**: Build on Mac, take simulator screenshots, feed them back to Claude Code.

1. Build and run on Mac (or via SSH)
2. Take screenshots of the running app
3. Share screenshots with Claude Code for visual verification
4. Claude Code can analyze screenshots and suggest UI fixes

For the Scaffluent app's sensor-dependent features (camera, thermal, ARKit), there is no substitute for real device testing. Focus Claude Code on the UI layer and business logic, and manually test hardware integrations.

### 4.4 Stale APIs → CLAUDE.md Rules + Version Pinning

**Solution**: Document preferred patterns in `CLAUDE.md`:

```markdown
## Swift/SwiftUI Conventions
- Deployment target: iOS 17+
- Use @Observable macro, NOT ObservableObject/@Published
- Use NavigationStack, NOT NavigationView
- Use async/await and @MainActor, NOT DispatchQueue.main.async
- Use .containerRelativeFrame() over GeometryReader where possible
- Use structured concurrency (TaskGroup, async let) over GCD
```

When Claude Code generates code with deprecated patterns, correct it once and add the correction to `CLAUDE.md`. It will follow the rule in subsequent interactions.

### 4.5 Hardware SDK Gaps → Provide SDK Documentation as Context

**Solution**: For the FLIR ONE Pro SDK and ARKit-specific code:

1. Copy relevant SDK header files or documentation into the project (or a `docs/` folder)
2. When asking Claude Code to modify `ThermalSDKWrapper` or `CameraService`, explicitly reference these docs
3. Keep hardware integration changes small and focused
4. Test hardware changes on real devices before expanding scope

### 4.6 Asset Catalog → Manual Management + Constants

**Solution**: Manage `Assets.xcassets` manually in Xcode. Use Claude Code to:
- Define color/font constants in `Utils/Constants.swift` referencing asset names
- Generate `Color("assetName")` and `Image("assetName")` references
- Keep a documented list of asset names in `CLAUDE.md`

### 4.7 Code Signing → Leave to Xcode

**Solution**: Code signing is fully managed by Xcode's automatic signing. No action needed from Claude Code. Add to `CLAUDE.md`:
```
Do not modify Info.plist signing or entitlements settings.
```

---

## 5. Recommended Workflow for the Scaffluent App <a name="recommended-workflow"></a>

### 5.1 Setup: Create a CLAUDE.md

Create a `CLAUDE.md` file in the `Scaffluent/` directory. This is loaded automatically at the start of every Claude Code session.

```markdown
# Scaffluent iOS App

## Architecture
- SwiftUI + Combine, MVVM pattern
- iOS 17+ deployment target
- Models → Services → ViewModels → Views

## Project Structure
- Models/: Data types (User, Issue, EvaluationSession, EvaluationResult, ModuleEnums)
- Services/: AuthService, SessionStore, CameraService, ThermalSDKWrapper, PermissionService
- Screens/{Feature}/: FeatureView.swift + FeatureViewModel.swift
- Components/: Reusable UI (PrimaryButton, StatusChipView, IssueRowView, SessionCardView)
- Modules/Aggregation/: Aggregator.swift (computes final results from issues)
- Utils/: Constants.swift, Logging.swift

## Rules
- NEVER modify .pbxproj, .xcodeproj, or .xcworkspace files
- Use @Observable (not ObservableObject/@Published) for new ViewModels
- Use NavigationStack (not NavigationView)
- Use async/await + @MainActor (not DispatchQueue.main.async)
- Use swift-log Logger for debug output
- Follow existing naming: {Feature}View.swift + {Feature}ViewModel.swift
- Keep views declarative; business logic goes in ViewModels
- Hardware SDK code (ThermalSDKWrapper, CameraService) changes need real device testing

## Build
- Open Scaffluent.xcodeproj in Xcode
- Scheme: Scaffluent
- Cannot build on Linux — macOS + Xcode required
```

### 5.2 Daily Development Loop

```
┌─────────────────────────────────────┐
│  1. Plan feature in Claude Code     │
│     (describe what you want)        │
├─────────────────────────────────────┤
│  2. Claude Code writes/edits Swift  │
│     files across the codebase       │
├─────────────────────────────────────┤
│  3. Open Xcode, add new files if    │
│     needed (or use folder refs)     │
├─────────────────────────────────────┤
│  4. Build in Xcode (Cmd+B)         │
│     Fix any build errors with       │
│     Claude Code (paste errors)      │
├─────────────────────────────────────┤
│  5. Run in Simulator/device         │
│     Screenshot any issues           │
├─────────────────────────────────────┤
│  6. Feed screenshots + observations │
│     back to Claude Code for fixes   │
├─────────────────────────────────────┤
│  7. Commit when feature works       │
└─────────────────────────────────────┘
```

### 5.3 Task Categories by Effectiveness

| Task | Claude Code Effectiveness | Notes |
|------|--------------------------|-------|
| New SwiftUI screen (View + ViewModel) | Very High | Follow existing patterns in Screens/ |
| New model type | Very High | Codable + Identifiable boilerplate |
| Refactor/rename across files | Very High | Cross-file awareness is key strength |
| Add business logic to ViewModel | High | Can reason about data flow |
| Create reusable Component | High | Follows existing Components/ patterns |
| Write unit tests | High | Can mock services and test ViewModels |
| Debug from error messages | High | Paste Xcode build errors directly |
| Modify Aggregator logic | High | Pure logic, no hardware dependency |
| Update AuthService/SessionStore | Medium-High | Pure logic, but test on device |
| Modify CameraService | Medium | ARKit/AVFoundation complexity |
| Modify ThermalSDKWrapper | Low-Medium | Niche SDK, needs device testing |
| Add assets/icons | Low | Manual Xcode work |
| Fix Xcode build settings | Not applicable | Don't modify project files |

### 5.4 Feature Development Example

**Adding a "Report Export" feature to the Scaffluent app:**

1. **Prompt Claude Code**:
   > "Add a PDF export feature to ResultsView. Create a new ReportService in Services/ that takes an EvaluationResult and generates a PDF using UIKit's PDF rendering. Add an 'Export Report' button to ResultsView that calls this service and presents a share sheet."

2. **Claude Code generates**:
   - `Services/ReportService.swift` — PDF generation logic
   - Edits to `Screens/Results/ResultsView.swift` — new button + share sheet
   - Possibly `Models/EvaluationResult.swift` — if additional computed properties needed

3. **Developer**:
   - Adds `ReportService.swift` to Xcode project (if not using folder refs)
   - Builds, tests, verifies PDF output
   - Feeds any issues back to Claude Code

### 5.5 Leveraging Claude Code for the Specific Scaffluent Modules

**Vision Module**: Claude Code can help design the detection pipeline interface, create mock data for testing, and build the UI overlay for displaying detections on camera frames.

**Thermal Module**: Keep `ThermalSDKWrapper` changes minimal via Claude Code. Instead, have Claude Code build the UI and data layer around it — displaying thermal data, computing temperature thresholds in the Aggregator, and formatting thermal issues in `IssueRowView`.

**Aggregation Module**: The `Aggregator` is pure logic — perfect for Claude Code. Ask it to implement scoring algorithms, severity weighting, and pass/fail thresholds.

---

## 6. Prompt Engineering Tips for Swift/SwiftUI <a name="prompt-engineering-tips"></a>

### 6.1 Be Explicit About iOS Version and Patterns

Bad: "Create a login screen"

Good: "Create LoginView.swift and LoginViewModel.swift in Screens/Login/ following the existing MVVM pattern. Use @Observable for the ViewModel. Include text fields for name and employee ID, a Picker for role selection (Instructor/Trainee), and a PrimaryButton. Validate that all fields are non-empty before enabling the button. On login, call AuthService.saveUser() and update AppViewModel.isLoggedIn."

### 6.2 Reference Existing Files

Bad: "Add a new screen for settings"

Good: "Create a SettingsView following the same pattern as Screens/Home/HomeView.swift. Use the same navigation style and PrimaryButton component. Include options for: clearing session history (via SessionStore), logging out (via AuthService), and toggling debug logging (via Constants)."

### 6.3 Paste Build Errors Directly

When Xcode fails to build after Claude Code's changes, copy the exact error messages:

```
"Here are the build errors after your changes:

Screens/Settings/SettingsView.swift:23:15 - Cannot find 'SessionStore' in scope
Services/AuthService.swift:45:9 - Missing return in a function expected to return 'User?'

Fix these errors."
```

### 6.4 Use "Think Step by Step" for Complex Features

For multi-component features, ask Claude Code to plan first:

> "I want to add offline support to the Scaffluent app so evaluations can be saved locally and synced when back online. Before writing code, analyze the current SessionStore implementation and outline what changes are needed across the codebase."

### 6.5 Constrain Scope Explicitly

> "ONLY modify ResultsView.swift. Do not change any models or services. Add a share button that shares the evaluation summary as plain text."

### 6.6 Request Incremental Changes

Instead of asking for a complete feature at once:

1. "Create the data model for Report" → review
2. "Create ReportService with a generatePDF method" → review
3. "Wire up the Export button in ResultsView" → review

This prevents large, hard-to-review changesets.

### 6.7 Provide Context for Hardware-Specific Code

> "The ThermalSDKWrapper receives frames from the FLIR ONE Pro SDK via a delegate callback `didReceiveFrame(_ frame: FLIRThermalImage)`. The frame has a `getValues()` method returning a 2D array of temperature readings in Celsius. Modify the thermal analysis to flag any reading above 60C as a Critical issue."

### 6.8 Ask for Previews

> "Add a SwiftUI Preview to the new SettingsView with mock data so I can verify the layout in Xcode Previews."

This is especially valuable since Claude Code can't see the rendered output — previews make it easy for you to verify visually.

---

## 7. Appendix: Scaffluent App Architecture Reference <a name="appendix"></a>

### File Structure

```
Scaffluent/
├── Scaffluent.xcodeproj/
├── Scaffluent/
│   ├── ScaffluentApp.swift              (app entry point)
│   ├── ContentView.swift                (root view / navigation)
│   ├── App/
│   │   ├── ScaffluentApp.swift          (scene setup)
│   │   └── AppViewModel.swift           (global state: user, session)
│   ├── Models/
│   │   ├── User.swift                   (user identity + role)
│   │   ├── Issue.swift                  (detected problem)
│   │   ├── EvaluationSession.swift      (session metadata)
│   │   ├── EvaluationResult.swift       (aggregated outcome)
│   │   └── ModuleEnums.swift            (ModuleSource, IssueSeverity)
│   ├── Screens/
│   │   ├── Login/                       (LoginView + LoginViewModel)
│   │   ├── Home/                        (HomeView — main hub)
│   │   ├── NewEvaluation/               (form to start session)
│   │   ├── Evaluation/                  (live inspection + sensors)
│   │   ├── Results/                     (final assessment)
│   │   ├── History/                     (past sessions list + detail)
│   │   └── Help/                        (documentation)
│   ├── Components/
│   │   ├── PrimaryButton.swift          (standard CTA button)
│   │   ├── StatusChipView.swift         (module status pill)
│   │   ├── IssueRowView.swift           (issue display row)
│   │   └── SessionCardView.swift        (history list card)
│   ├── Services/
│   │   ├── AuthService.swift            (user persistence)
│   │   ├── SessionStore.swift           (session CRUD)
│   │   ├── CameraService.swift          (AVFoundation + ARKit)
│   │   ├── ThermalSDKWrapper.swift      (FLIR ONE Pro)
│   │   └── PermissionService.swift      (camera/location perms)
│   ├── Modules/
│   │   └── Aggregation/
│   │       └── Aggregator.swift         (issues → final result)
│   └── Utils/
│       ├── Constants.swift              (app-wide constants)
│       └── Logging.swift                (logging setup)
```

### Data Flow

```
User Login → AuthService → AppViewModel.isLoggedIn
                                    ↓
                               HomeView
                                    ↓
                          NewEvaluationView
                                    ↓
                    SessionStore.createSession()
                                    ↓
                          EvaluationView
                                    ↓
              EvaluationViewModel orchestrates:
              ├── CameraService → Vision Module → [Issue]
              ├── CameraService → Tilt Module → [Issue]
              └── ThermalSDKWrapper → Thermal Module → [Issue]
                                    ↓
                    Aggregator([Issue]) → EvaluationResult
                                    ↓
                            ResultsView
                                    ↓
                    SessionStore.save(session, result)
                                    ↓
                    HistoryListView → SessionDetailView
```

### Key Integration Points for Claude Code

When asking Claude Code to modify the app, these are the most productive entry points:

1. **New screens**: Create in `Screens/{Feature}/` with View + ViewModel pair
2. **New models**: Add to `Models/`, update consumers
3. **Business logic**: Modify ViewModels or Aggregator
4. **UI polish**: Modify Components/ or individual Views
5. **Data persistence**: Modify SessionStore or AuthService
6. **Tests**: Create alongside ViewModels and Services

---

*Report generated 2026-03-05. Based on analysis of the Scaffluent iOS app codebase and research into Claude Code capabilities for Swift/iOS development.*
