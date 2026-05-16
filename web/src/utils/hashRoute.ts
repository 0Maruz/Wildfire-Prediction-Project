import { useEffect, useState } from "react";
import type { AlertPageRoute } from "../types";

// Tiny hash-router (no React Router dependency). URL `#notify` → "notify",
// `#reports` → "reports", default → "dashboard". Updating the route writes
// to location.hash so browser back/forward + share-link work naturally.

const VALID: AlertPageRoute[] = ["dashboard", "notify", "reports"];

export function getRoute(): AlertPageRoute {
  const hash = window.location.hash.replace(/^#\/?/, "").toLowerCase();
  if (VALID.includes(hash as AlertPageRoute)) {
    return hash as AlertPageRoute;
  }
  return "dashboard";
}

export function useHashRoute(): [AlertPageRoute, (r: AlertPageRoute) => void] {
  const [route, setRoute] = useState<AlertPageRoute>(getRoute);
  useEffect(() => {
    const onChange = () => setRoute(getRoute());
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  const navigate = (r: AlertPageRoute) => {
    window.location.hash = r === "dashboard" ? "" : `#${r}`;
  };
  return [route, navigate];
}
