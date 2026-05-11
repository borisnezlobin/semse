import AppKit
import SwiftUI

struct SearchView: View {
    let onDismiss: () -> Void
    let onContentHeightChange: (CGFloat) -> Void

    @State private var query = ""
    @State private var response: SearchResponse?
    @State private var loading = false
    @State private var error: String?
    @State private var highlightIdx = 0

    @State private var debounceTask: Task<Void, Never>?
    @State private var currentTask: Task<Void, Never>?
    @State private var idleTask: Task<Void, Never>?
    @FocusState private var focused: Bool

    private let minQueryLength = 3
    private let debounceMs: UInt64 = 250
    private let idleResetSeconds: UInt64 = 60
    private let scrollMaxHeight: CGFloat = 560

    var body: some View {
        VStack(spacing: 0) {
            searchInput
                .padding(.horizontal, 16)
                .padding(.vertical, 16)

            if showsBody {
                Divider().background(Theme.dividerLine)
                // No SwiftUI ScrollView — it interacts poorly with NSHostingView's
                // intrinsic sizing (the SwiftUI scroll machinery proposes its own
                // height, which fights our KVO-driven panel resize). Instead, the
                // content lays out naturally; the AppKit panel grows up to its max
                // height, and we use an AppKit NSScrollView wrapper *outside*
                // SwiftUI when the content overflows.
                resultsContent
                    .padding(.horizontal, 12)
                    .padding(.vertical, 10)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .frame(width: 720)
        .background(
            GeometryReader { proxy in
                Color.clear.preference(key: ContentHeightKey.self, value: proxy.size.height)
            }
        )
        .onPreferenceChange(ContentHeightKey.self) { newHeight in
            onContentHeightChange(newHeight)
        }
        .onAppear { focused = true; armIdleReset() }
        .onReceive(NotificationCenter.default.publisher(for: .focusSearchField)) { _ in
            focused = true
            armIdleReset()
        }
        .background(KeyMonitor(onEsc: handleEsc, onArrow: handleArrow))
    }

    private var resultsContent: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let err = error {
                Text(err)
                    .font(.system(size: 12.5))
                    .foregroundColor(.red.opacity(0.85))
                    .padding(.horizontal, 4)
            }
            if loading && response == nil {
                ShimmerCards()
            }
            if let answer = response?.answer, !answer.isEmpty {
                SynthesisCard(answer: answer)
            }
            if let sources = response?.sources, !sources.isEmpty {
                ForEach(Array(sources.enumerated()), id: \.offset) { idx, src in
                    if src.source == "image" {
                        ImageResultCard(result: src, highlighted: idx == highlightIdx)
                            .onHover { hovering in if hovering { highlightIdx = idx } }
                    } else {
                        SourceCard(result: src, highlighted: idx == highlightIdx)
                            .onHover { hovering in if hovering { highlightIdx = idx } }
                    }
                }
            } else if let r = response, r.sources.isEmpty, !loading {
                emptyResults
            }
            if let r = response {
                footer(r)
            }
        }
    }

    private var showsBody: Bool {
        loading || response != nil || error != nil
    }

    private var searchInput: some View {
        HStack(spacing: 12) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 16, weight: .light))
                .foregroundColor(Theme.tertiaryText.opacity(0.9))
            ZStack(alignment: .leading) {
                if query.isEmpty {
                    Text("Search your conversations…")
                        .font(.system(size: 17))
                        .foregroundColor(Theme.placeholder)
                }
                TextField("", text: $query)
                    .textFieldStyle(.plain)
                    .font(.system(size: 17))
                    .foregroundColor(.white)
                    .focused($focused)
                    .onChange(of: query) { _, newValue in
                        scheduleSearch(for: newValue)
                        armIdleReset()
                    }
            }
        }
    }

    private var emptyResults: some View {
        Text("No matches.")
            .font(.system(size: 12.5))
            .foregroundColor(Theme.tertiaryText)
            .frame(maxWidth: .infinity, alignment: .center)
            .padding(.vertical, 20)
    }

    private func footer(_ r: SearchResponse) -> some View {
        HStack {
            Spacer()
            Text("\(r.sources.count) source\(r.sources.count == 1 ? "" : "s") · \(r.queryMs)ms")
                .font(.system(size: 10.5))
                .foregroundColor(Theme.tertiaryText.opacity(0.7))
        }
        .padding(.top, 4)
        .padding(.horizontal, 4)
    }

    // MARK: - Actions

    private func scheduleSearch(for value: String) {
        debounceTask?.cancel()
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.count < minQueryLength {
            response = nil
            error = nil
            loading = false
            currentTask?.cancel()
            return
        }
        debounceTask = Task {
            try? await Task.sleep(nanoseconds: debounceMs * 1_000_000)
            if Task.isCancelled { return }
            await runSearch(trimmed)
        }
    }

    private func runSearch(_ q: String) async {
        currentTask?.cancel()
        let task = Task {
            await MainActor.run { loading = true; error = nil }
            do {
                let resp = try await SearchClient.shared.search(query: q)
                if Task.isCancelled { return }
                await MainActor.run {
                    response = resp
                    loading = false
                    highlightIdx = 0
                }
            } catch {
                if Task.isCancelled { return }
                await MainActor.run {
                    self.error = error.localizedDescription
                    self.loading = false
                }
            }
        }
        currentTask = task
        await task.value
    }

    private func armIdleReset() {
        idleTask?.cancel()
        idleTask = Task {
            try? await Task.sleep(nanoseconds: idleResetSeconds * 1_000_000_000)
            if Task.isCancelled { return }
            await MainActor.run {
                query = ""
                response = nil
                error = nil
            }
        }
    }

    private func handleEsc() {
        if !query.isEmpty || response != nil {
            query = ""
            response = nil
            error = nil
        } else {
            onDismiss()
        }
    }

    private func handleArrow(_ dir: ArrowDirection) {
        guard let count = response?.sources.count, count > 0 else { return }
        switch dir {
        case .down: highlightIdx = min(count - 1, highlightIdx + 1)
        case .up:   highlightIdx = max(0, highlightIdx - 1)
        }
        armIdleReset()
    }
}

enum ArrowDirection { case up, down }

extension Notification.Name {
    static let focusSearchField = Notification.Name("memory.focusSearchField")
}

/// SearchView reports its laid-out height through this preference so the
/// AppKit panel can resize. Replaces the old NSHostingView.intrinsicContentSize
/// KVO, which doesn't fire on SwiftUI re-layouts.
private struct ContentHeightKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}

/// Catches Esc and arrow keys at the window level so they work even when the
/// text field is focused.
private struct KeyMonitor: NSViewRepresentable {
    let onEsc: () -> Void
    let onArrow: (ArrowDirection) -> Void

    func makeNSView(context: Context) -> NSView {
        let view = MonitorView()
        view.onEsc = onEsc
        view.onArrow = onArrow
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {}

    final class MonitorView: NSView {
        var onEsc: (() -> Void)?
        var onArrow: ((ArrowDirection) -> Void)?
        private var monitor: Any?

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            monitor.map { NSEvent.removeMonitor($0) }
            guard window != nil else { return }
            monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
                guard let self = self else { return event }
                switch event.keyCode {
                case 53: self.onEsc?(); return nil          // esc
                case 125: self.onArrow?(.down); return nil  // arrow down
                case 126: self.onArrow?(.up); return nil    // arrow up
                default: return event
                }
            }
        }

        deinit { monitor.map { NSEvent.removeMonitor($0) } }
    }
}

struct ShimmerCards: View {
    @State private var phase: CGFloat = -1

    var body: some View {
        VStack(spacing: 8) {
            ForEach(0..<3) { _ in
                shimmerRow
            }
        }
        .onAppear {
            withAnimation(.linear(duration: 1.4).repeatForever(autoreverses: false)) {
                phase = 1
            }
        }
    }

    private var shimmerRow: some View {
        HStack(spacing: 10) {
            Circle().fill(Color.white.opacity(0.06)).frame(width: 28, height: 28)
            VStack(alignment: .leading, spacing: 6) {
                RoundedRectangle(cornerRadius: 4).fill(Color.white.opacity(0.06)).frame(height: 10).frame(maxWidth: 140)
                RoundedRectangle(cornerRadius: 4).fill(Color.white.opacity(0.06)).frame(height: 10)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(RoundedRectangle(cornerRadius: 12).fill(Color.white.opacity(0.02)))
    }
}
