---
name: Precision Optics Consumer Tech
colors:
  surface: '#f7f9ff'
  surface-dim: '#d3dbe5'
  surface-bright: '#f7f9ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#ecf4ff'
  surface-container: '#e7eff9'
  surface-container-high: '#e1e9f3'
  surface-container-highest: '#dbe3ed'
  on-surface: '#141c23'
  on-surface-variant: '#42474f'
  inverse-surface: '#293139'
  inverse-on-surface: '#eaf1fc'
  outline: '#72777f'
  outline-variant: '#c2c7d0'
  surface-tint: '#35618d'
  primary: '#00375e'
  on-primary: '#ffffff'
  primary-container: '#1f4e79'
  on-primary-container: '#95bff1'
  inverse-primary: '#a0cafc'
  secondary: '#835500'
  on-secondary: '#ffffff'
  secondary-container: '#feae2c'
  on-secondary-container: '#6b4500'
  tertiary: '#2e353b'
  on-tertiary: '#ffffff'
  tertiary-container: '#454c52'
  on-tertiary-container: '#b5bcc3'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#d1e4ff'
  primary-fixed-dim: '#a0cafc'
  on-primary-fixed: '#001d35'
  on-primary-fixed-variant: '#184974'
  secondary-fixed: '#ffddb4'
  secondary-fixed-dim: '#ffb955'
  on-secondary-fixed: '#291800'
  on-secondary-fixed-variant: '#633f00'
  tertiary-fixed: '#dce3ea'
  tertiary-fixed-dim: '#c0c7ce'
  on-tertiary-fixed: '#151c21'
  on-tertiary-fixed-variant: '#40484d'
  background: '#f7f9ff'
  on-background: '#141c23'
  surface-variant: '#dbe3ed'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 56px
    fontWeight: '600'
    lineHeight: 64px
    letterSpacing: -0.03em
  display-lg-mobile:
    fontFamily: Inter
    fontSize: 38px
    fontWeight: '600'
    lineHeight: 44px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 36px
    fontWeight: '600'
    lineHeight: 44px
    letterSpacing: -0.02em
  headline-lg-mobile:
    fontFamily: Inter
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 34px
    letterSpacing: -0.015em
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '500'
    lineHeight: 28px
    letterSpacing: -0.005em
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
    letterSpacing: -0.005em
  body-md:
    fontFamily: Inter
    fontSize: 15px
    fontWeight: '400'
    lineHeight: 22px
    letterSpacing: 0em
  body-sm:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
    letterSpacing: 0.005em
  label-lg:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '600'
    lineHeight: 20px
    letterSpacing: 0.01em
  label-md:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.02em
  label-technical:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: '600'
    lineHeight: 14px
    letterSpacing: 0.06em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  gutter: 1.5rem
  gutter-mobile: 1rem
  margin: 3rem
  margin-mobile: 1.25rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 1rem
  space-lg: 1.5rem
  space-xl: 2.5rem
---

## Brand & Style

The design system embodies the calculated precision, engineering restraint, and tactile elegance of high-end industrial design (evoking Apple and Dyson). It is crafted for modern homeowners and tech enthusiasts managing an autonomous, ultra-slim pocket robot navigating low-clearance domestic spaces.

The visual narrative bridges physical robotics with frictionless digital control. The aesthetic merges **Minimalism** with subtle **Tactile Precision**:
- **Purity and Precision:** Vast negative space, strict geometric alignments, and architectural light gray-blue backdrops prevent visual clutter, establishing immediate technological authority.
- **Physicality and Utility:** Crisp border contours, laser-focused search beams, and warm amber optical highlights mirror physical LIDAR/sensor beacons and low-light search activities under furniture.
- **Tone:** Methodical, premium, calm, reassuring, and exceptionally clean.

## Colors

The palette balances authoritative engineering tone with high-visibility kinetic alerts:

- **Primary (`#1F4E79` - Deep Navy):** Represents core structural components, high-priority navigation bars, main headings, and primary interface commands. It grounds the product with architectural stability.
- **Secondary (`#F5A623` - Warm Amber):** The luminous accent signifying the robot's optical scanning beam, active search status, beacon highlights, and focal calls-to-action.
- **Tertiary (`#EAF1F8` - Light Gray-Blue):** Applied to full-width canvas alternations, module containers, and inset surfaces, offering soft tonal distinction against pure white without visual noise.
- **Neutral (`#121A21` - Midnight Slate):** Precision body typography and ultra-dark vector iconography, ensuring strict WCAG AAA contrast ratios.
- **Structural Line (`#D6E4F0` - Arctic Stroke):** Delicate, architectural boundary line defining cards, data tables, and interactive modules.
- **Surface Pure (`#FFFFFF`):** Reserved for elevated cards, active panels, and floating controls to provide crisp physical detachment from `#EAF1F8` backdrops.

## Typography

The typography scale relies on **Inter** (supported smoothly by Pretendard or system Apple fonts as local fallbacks). It adheres to strict consumer hardware display metrics:

- **Negative Tracking on Large Scales:** Display and Headline roles utilize tight tracking (`-0.01em` to `-0.03em`) to mimic machined product engravings and crisp editorial headlines.
- **Telemetry & Technical Microcopy:** `label-technical` is rendered in uppercase with wide letter-spacing (`0.06em`) for battery levels, millimeter clearance metrics, and LIDAR frequency labels.
- **Reading Comfort:** Body text preserves generous line heights (`1.45` to `1.55`) to facilitate immediate comprehension during live robot search cycles.

## Layout & Spacing

The layout philosophy leverages an adaptive fluid grid with generous mathematical gutters that allow complex telemetry readouts to breathe.

- **Grid Architecture:** 
  - **Desktop (1200px+):** 12-column layout, 24px gutters, max-width container capped at 1280px with minimum 48px outer margins.
  - **Tablet (768px - 1199px):** 8-column layout, 20px gutters, 32px outer margins.
  - **Mobile (< 768px):** 4-column layout, 16px gutters, 20px outer margins.
- **Section Pacing:** Alternating structural bands alternate seamlessly between `#FFFFFF` and `#EAF1F8`, separated by minimum `4rem` vertical breathing room to reflect clean consumer hardware marketing.
- **In-Component Rhythm:** Strict 4px/8px incremental spacing scale ensures interior component alignment across radar cards, status telemetry, and interactive controls.

## Elevation & Depth

Visual hierarchy is maintained through crisp structural contours paired with luminous, diffused ambient drop shadows rather than heavy skeuomorphic shading.

- **Level 0 (Flat / Recessed):** Ground surfaces and data pits using `#EAF1F8` with a 1px uniform border in `#D6E4F0`.
- **Level 1 (Card / Rest):** Elevated pure white (`#FFFFFF`) containers resting on `#EAF1F8`. Surrounded by a 1px border (`#D6E4F0`) and supported by an ultra-subtle, deep-navy tinted ambient shadow: `box-shadow: 0 4px 20px -2px rgba(31, 78, 121, 0.05)`.
- **Level 2 (Interactive Hover / Floating Panels):** Floating navigation bars, active item targeting cards, and sensor tooltips: `box-shadow: 0 12px 32px -4px rgba(31, 78, 121, 0.08), 0 2px 6px -1px rgba(31, 78, 121, 0.03)`.
- **Level 3 (Modal / Real-Time Tracking HUD):** Focused search beam overlays and manual joystick panels: `box-shadow: 0 24px 48px -8px rgba(31, 78, 121, 0.12)`.
- **Search Beam Glow:** Dynamic search states leverage an amber atmospheric spread: `box-shadow: 0 0 24px rgba(245, 166, 35, 0.25)`.

## Shapes

The interface mirrors the smooth, pocket-friendly industrial curvature of the robot hardware:

- **Base Radius (16px / `rounded-lg`):** Standard for all content cards, sensor video feeds, scanning maps, and modal viewports.
- **Interactive Small (8px / `rounded`):** Applied to form inputs, segmented control segments, dropdown lists, and telemetry pills.
- **Full Pill (`rounded-full`):** Reserved strictly for primary action buttons, beacon indicators, battery indicators, and live ping tags.

## Components

### Buttons
- **Primary Beam CTA:** Warm amber (`#F5A623`) background, dark slate (`#121A21`) text, pill-shaped (`rounded-full`), padded `12px 28px`. Hover increases saturation with an amber optical glow (`0 0 16px rgba(245,166,35,0.4)`).
- **Secondary System CTA:** Deep navy (`#1F4E79`) fill with `#FFFFFF` text, subtle elevation (Level 1).
- **Ghost / Outline Button:** 1px border (`#D6E4F0`), transparent surface, deep navy typography. Fills with `#EAF1F8` on hover.

### Cards & Modules
- Pure white (`#FFFFFF`) surface with exactly 16px corner radius (`rounded-lg`), 1px continuous border (`#D6E4F0`), and Level 1 ambient navy shadow.
- Interior padding set to `24px` (`space-lg`). Header metadata separated by hairline `#D6E4F0` division when presenting telemetry tables.

### Input Fields & Controls
- **Inputs:** Crisp white or light `#EAF1F8` background, 8px corner radius, 1px border in `#D6E4F0`, body-md text. Active focus states replace the border with Deep Navy (`#1F4E79`) and a 3px soft outer ring (`rgba(31, 78, 121, 0.15)`).
- **Checkboxes & Radios:** 20px precision rings with 1.5px `#D6E4F0` stroke. Checked state transitions smoothly to `#1F4E79` with crisp micro-white checks. Active locator toggles use `#F5A623` when arming real-time acoustic pings.

### Chips & Telemetry Tags
- Compact 24px height, full pill radius, `label-technical` typography.
- Neutral state: `#EAF1F8` background, `#1F4E79` text.
- Active item scanning: Amber tint (`rgba(245, 166, 35, 0.15)`) with `#D98200` text and a pulsing 6px amber dot.

### Specialized Robot HUD Components
- **Under-Furniture Clearance Gauge:** Horizontal miniature progress gauge featuring a 4mm visual ceiling marker, filled with Deep Navy and accented with Amber beacon markers.
- **LIDAR Acoustic Ping Radar:** Concentric 1px circles rendered in `#D6E4F0` over a soft `#EAF1F8` inset, with animated radial sweep in `#F5A623`.