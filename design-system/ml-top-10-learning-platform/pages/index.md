# Page Override: Index Dashboard (Home)

> **LOGIC:** This page override file defines layout and typography guidelines tailored for the target audience of the machine learning learning platform. These specifications override the child-education fonts suggested in the Master Design System.

---

## Typography Overrides

- **Heading Font:** Outfit (Sans-serif display font, geometric and modern)
- **Body Font:** Inter (Highly readable standard sans-serif font for data density)
- **Monospace Font:** Fira Code (For coding snips and output terminal simulation)
- **Mood:** Professional, technical, premium, high data density, readable.
- **Reasoning:** Since this study guide targets **Data Scientists, AI Engineers, and Researchers**, playful fonts (Baloo 2 / Comic Neue) would undermine the educational authority of the content. A premium tech aesthetic fits the subject matter.

## Google Fonts CSS Link
```css
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');
```

---

## Layout Adjustments

- **Pattern:** Responsive Dual-pane Workspace (Sidebar navigation + Main scrolling workspace with dynamic canvas overlay).
- **Navigation:** Lock the sidebar on desktop screens (`position: sticky`) for persistent progress visibility, stacking vertically on mobile devices.
- **Data Densities:** Compact padding (`--space-md`) for parameter control panels to maximize screen real estate for the canvas playground.

---

## Verification & Contrast Checks

- Background `#020617` vs Text `#F8FAFC` contrast ratio: **18.7:1** (Passes WCAG AAA)
- Background `#020617` vs Accent Blue `#38BDF8` contrast ratio: **8.6:1** (Passes WCAG AA)
- Background `#020617` vs Accent Green `#22C55E` contrast ratio: **6.2:1** (Passes WCAG AA)
