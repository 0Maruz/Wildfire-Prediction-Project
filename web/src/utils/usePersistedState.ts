import { useEffect, useState } from "react";

// Tiny localStorage-backed state. Returns the persisted value (falling back
// to `initial` if missing/invalid) and a setter that writes through to
// localStorage. Quietly swallows storage exceptions (private mode, etc).

export function usePersistedState<T>(
  key: string,
  initial: T,
): [T, (v: T | ((prev: T) => T)) => void] {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = window.localStorage.getItem(key);
      if (raw == null) return initial;
      return JSON.parse(raw) as T;
    } catch {
      return initial;
    }
  });

  useEffect(() => {
    try {
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch {
      /* private mode / storage disabled — no-op */
    }
  }, [key, value]);

  return [value, setValue];
}
