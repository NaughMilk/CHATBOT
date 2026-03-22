import { useEffect, useState, useRef } from "react";
import API_URL from "../config";

const POLL_INTERVAL = 5000; // poll every 5s
const TIPS = [
  "Server miễn phí sẽ tạm ngủ khi không có ai truy cập...",
  "Đang đánh thức server, vui lòng đợi trong giây lát ☕",
  "Server đang khởi động lại mô hình AI...",
  "Quá trình này có thể mất 1–3 phút, hãy kiên nhẫn nhé 🙏",
  "Tip: Truy cập thường xuyên để server không bị ngủ!",
];

export default function ServerStatus({ children }) {
  const [serverReady, setServerReady] = useState(false);
  const [checking, setChecking] = useState(true);
  const [elapsed, setElapsed] = useState(0);
  const [tipIndex, setTipIndex] = useState(0);
  const startRef = useRef(Date.now());

  useEffect(() => {
    let alive = true;

    const check = async () => {
      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 8000);
        const res = await fetch(`${API_URL}/health`, {
          signal: controller.signal,
        });
        clearTimeout(timeout);
        if (res.ok && alive) {
          setServerReady(true);
          setChecking(false);
        }
      } catch {
        // server still cold
      }
    };

    check();
    const interval = setInterval(() => {
      if (!alive) return;
      check();
    }, POLL_INTERVAL);

    return () => {
      alive = false;
      clearInterval(interval);
    };
  }, []);

  // Elapsed timer
  useEffect(() => {
    if (serverReady) return;
    const timer = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startRef.current) / 1000));
    }, 1000);
    return () => clearInterval(timer);
  }, [serverReady]);

  // Rotate tips
  useEffect(() => {
    if (serverReady) return;
    const timer = setInterval(() => {
      setTipIndex((prev) => (prev + 1) % TIPS.length);
    }, 6000);
    return () => clearInterval(timer);
  }, [serverReady]);

  if (serverReady) return children;

  const mins = Math.floor(elapsed / 60);
  const secs = elapsed % 60;
  const timeStr = mins > 0 ? `${mins}:${String(secs).padStart(2, "0")}` : `${secs}s`;

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-ink">
      <div className="flex flex-col items-center gap-8 px-6 text-center max-w-md">
        {/* Animated logo / spinner */}
        <div className="relative">
          <div className="h-20 w-20 rounded-full border-4 border-white/10 border-t-tide animate-spin" />
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="h-10 w-10 rounded-full bg-tide/20 animate-pulse" />
          </div>
        </div>

        {/* Title */}
        <div>
          <h2 className="text-xl font-semibold text-fog">
            Đang khởi động server
          </h2>
          <p className="mt-2 text-sm text-haze">
            Server đang được đánh thức từ chế độ nghỉ
          </p>
        </div>

        {/* Progress bar (fake but reassuring) */}
        <div className="w-full">
          <div className="h-1.5 w-full rounded-full bg-white/10 overflow-hidden">
            <div
              className="h-full rounded-full bg-gradient-to-r from-tide to-pine transition-all duration-1000 ease-out"
              style={{ width: `${Math.min(95, elapsed * 0.8)}%` }}
            />
          </div>
        </div>

        {/* Rotating tip */}
        <p className="text-sm text-haze transition-opacity duration-500 min-h-[40px]">
          {TIPS[tipIndex]}
        </p>

        {/* Timer */}
        <div className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs text-haze">
          Đã chờ: {timeStr}
        </div>
      </div>
    </div>
  );
}
