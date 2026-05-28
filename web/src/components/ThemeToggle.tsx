import { useEffect, useState } from "react";
import { fmtTr, useLang } from "../utils/i18n";

// Light/dark theme toggle. Persists choice in localStorage and respects the
// user's OS preference on first visit (prefers-color-scheme).
//
// Implementation: toggles a `data-theme="light"` attribute on <html>, which
// the CSS at the bottom of styles.css overrides --bg / --surface / --text
// custom properties under that selector. Default = dark.

type Theme = "dark" | "light";

const STORAGE_KEY = "firewatch.theme";

function detectInitial(): Theme {
  if (typeof window === "undefined") return "dark";
  const stored = window.localStorage.getItem(STORAGE_KEY);
  if (stored === "dark" || stored === "light") return stored;
  // OS preference fallback
  if (window.matchMedia?.("(prefers-color-scheme: light)").matches) return "light";
  return "dark";
}

function apply(theme: Theme) {
  if (theme === "light") document.documentElement.setAttribute("data-theme", "light");
  else document.documentElement.removeAttribute("data-theme");
}

export default function ThemeToggle() {
  const { t } = useLang();
  const [theme, setTheme] = useState<Theme>(detectInitial);

  useEffect(() => {
    apply(theme);
    try {
      window.localStorage.setItem(STORAGE_KEY, theme);
    } catch {
      // private mode / disabled storage — silently fall back to session-only
    }
  }, [theme]);

  const next: Theme = theme === "dark" ? "light" : "dark";
  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={() => setTheme(next)}
      title={fmtTr(t("theme.switch"), { theme: next === "light" ? t("theme.light") : t("theme.dark") })}
      aria-label={fmtTr(t("theme.switch"), { theme: next === "light" ? t("theme.light") : t("theme.dark") })}
    >
      {theme === "dark" ? "🌙" : "☀️"}
    </button>
  );
}
