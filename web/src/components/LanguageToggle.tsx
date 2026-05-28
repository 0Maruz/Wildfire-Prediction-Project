import { useLang } from "../utils/i18n";

// Tiny EN/TH toggle that mirrors ThemeToggle's visual language. The state
// is owned by the LanguageContext provider at App root.
export default function LanguageToggle() {
  const { lang, setLang, t } = useLang();
  const next = lang === "en" ? "th" : "en";
  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={() => setLang(next)}
      title={t("lang.toggle.label", "Language")}
      aria-label={t("lang.toggle.label", "Language")}
    >
      <span style={{ fontSize: 12, fontWeight: 600 }}>
        {lang === "en" ? t("lang.toggle.en", "EN") : t("lang.toggle.th", "ไทย")}
      </span>
    </button>
  );
}
