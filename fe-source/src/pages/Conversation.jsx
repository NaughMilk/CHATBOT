import { Link, useNavigate } from "react-router-dom";
import { useEffect, useRef, useState } from "react";
import API_URL from "../config";

export default function Conversation() {
  const navigate = useNavigate();

  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [listening, setListening] = useState(false);
  const [locked, setLocked] = useState(false);

  // ====== CONFIG ======
  // Nếu bạn muốn mỗi lượt nói xong -> xoá toàn bộ chat UI (context trong khung chat)
  const CLEAR_CHAT_EACH_TURN = false;

  // ====== Refs ======
  const greetedRef = useRef(false);
  const doneRef = useRef(false);
  const listRef = useRef(null);

  const recogRef = useRef(null);
  const transcriptRef = useRef("");
  const finalTranscriptRef = useRef("");

  const lastSpokenRef = useRef("");
  const reqSeqRef = useRef(0);

  const isAgentSpeakingRef = useRef(false);
  const autoListenRef = useRef(true);

  const loadingRef = useRef(false);
  const lockedRef = useRef(false);
  const listeningRef = useRef(false);

  const onResultRef = useRef(null);
  const onEndRef = useRef(null);
  const onErrorRef = useRef(null);

  const sendMessageRef = useRef(null);
  const speechFallbackRef = useRef(null);
  const speechSeqRef = useRef(0);
  const activeSpeechSeqRef = useRef(0);
  const emptySpeechRetryRef = useRef(null);
  const heardSpeechRef = useRef(false);
  const silenceSendTimerRef = useRef(null);
  const resultOffsetRef = useRef(0);

  // State machine để tránh auto-loop sai
  // idle | listening | sending | speaking
  const phaseRef = useRef("idle");


  // ====== Helpers ======
  const clearSpeechFallback = () => {
    if (speechFallbackRef.current) {
      clearTimeout(speechFallbackRef.current);
      speechFallbackRef.current = null;
    }
  };

  const clearEmptySpeechRetry = () => {
    if (emptySpeechRetryRef.current) {
      clearTimeout(emptySpeechRetryRef.current);
      emptySpeechRetryRef.current = null;
    }
  };

  const clearSilenceSendTimer = () => {
    if (silenceSendTimerRef.current) {
      clearTimeout(silenceSendTimerRef.current);
      silenceSendTimerRef.current = null;
    }
  };

  const stopListening = () => {
    const recog = recogRef.current;
    if (!recog) return;

    try {
      recog.stop();
    } catch {
      // ignore
    }
    setListening(false);
    listeningRef.current = false;
    clearEmptySpeechRetry();
    clearSilenceSendTimer();

    if (phaseRef.current === "listening") {
      phaseRef.current = "idle";
    }
  };

  const startListening = () => {
    const recog = recogRef.current;

    if (lockedRef.current) return;
    if (loadingRef.current) return;
    if (isAgentSpeakingRef.current) return;
    if (!autoListenRef.current) return;

    if (!recog) {
      setMessages((prev) => [
        ...prev,
        { from: "coach", text: "Trình duyệt chưa hỗ trợ Speech API." },
      ]);
      return;
    }

    // Chỉ bật mic khi đang idle
    if (phaseRef.current !== "idle") return;
    if (listeningRef.current) return;

    // reset transcript buffers + input
    transcriptRef.current = "";
    finalTranscriptRef.current = "";
    resultOffsetRef.current = 0;
    setInput("");
    heardSpeechRef.current = false;
    clearEmptySpeechRetry();
    clearSilenceSendTimer();

    phaseRef.current = "listening";
    setListening(true);
    listeningRef.current = true;

    try {
      recog.start();
    } catch {
      // Nếu start lỗi (thường do gọi start quá nhanh)
      phaseRef.current = "idle";
      setListening(false);
      listeningRef.current = false;
    }
  };

  // ====== Backend TTS (Google Cloud TTS with SSML) ======
  const activeAudioRef = useRef(null);

  const speakText = (text, ssml) => {
    const t = (text || "").trim();
    if (!t) return false;

    if (lastSpokenRef.current === t) return false;
    lastSpokenRef.current = t;

    const speechSeq = speechSeqRef.current + 1;
    speechSeqRef.current = speechSeq;
    activeSpeechSeqRef.current = speechSeq;

    clearSpeechFallback();
    clearSilenceSendTimer();

    phaseRef.current = "speaking";
    isAgentSpeakingRef.current = true;

    stopListening();

    // Cancel any ongoing audio
    if (activeAudioRef.current) {
      activeAudioRef.current.pause();
      activeAudioRef.current = null;
    }

    const finishSpeaking = () => {
      if (activeSpeechSeqRef.current !== speechSeq) return;
      isAgentSpeakingRef.current = false;
      activeAudioRef.current = null;
      clearSpeechFallback();
      phaseRef.current = "idle";
      if (!lockedRef.current && autoListenRef.current) {
        startListening();
      }
    };

    // Send to backend POST /tts → get MP3 → play
    fetch(`${API_URL}/tts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: t,
        ssml: ssml || null,
        lang: "vi-VN",
        speaking_rate: 1.0,
      }),
    })
      .then((res) => {
        if (!res.ok) throw new Error(`TTS HTTP ${res.status}`);
        return res.blob();
      })
      .then((blob) => {
        if (activeSpeechSeqRef.current !== speechSeq) return;
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        activeAudioRef.current = audio;

        audio.onended = () => {
          if (activeSpeechSeqRef.current !== speechSeq) return;
          URL.revokeObjectURL(url);
          finishSpeaking();
        };

        audio.onerror = () => {
          if (activeSpeechSeqRef.current !== speechSeq) return;
          console.warn("[TTS] Audio playback error, finishing speaking");
          URL.revokeObjectURL(url);
          finishSpeaking();
        };

        audio.play().catch(() => {
          if (activeSpeechSeqRef.current !== speechSeq) return;
          console.warn("[TTS] Audio play() failed, finishing speaking");
          URL.revokeObjectURL(url);
          finishSpeaking();
        });
      })
      .catch((err) => {
        console.warn("[TTS] Fetch error:", err);
        if (activeSpeechSeqRef.current !== speechSeq) return;
        finishSpeaking();
      });

    // Fallback timer: if audio hasn't ended after approxMs, force finish
    const approxMs = Math.min(30000, Math.max(3000, t.length * 80));
    let retryLeft = 20;
    const fallbackTick = () => {
      if (activeSpeechSeqRef.current !== speechSeq) return;
      const audioBusy = activeAudioRef.current && !activeAudioRef.current.paused;
      if (audioBusy) {
        if (retryLeft > 0) {
          retryLeft -= 1;
          speechFallbackRef.current = setTimeout(fallbackTick, 500);
        } else {
          finishSpeaking();
        }
        return;
      }
      if (!lockedRef.current && autoListenRef.current && isAgentSpeakingRef.current) {
        finishSpeaking();
      }
    };
    speechFallbackRef.current = setTimeout(fallbackTick, approxMs);

    return true;
  };

  const sendMessage = async (overrideText) => {
    const userId = localStorage.getItem("user_id");
    const threadId = localStorage.getItem("thread_id");

    const rawText = typeof overrideText === "string" ? overrideText : input;
    const text = (rawText || "").trim();

    if (!text || !userId || !threadId || lockedRef.current || loadingRef.current) {
      return;
    }

    // user nói xong -> tắt mic + chuyển sending
    stopListening();
    phaseRef.current = "sending";
    clearSilenceSendTimer();

    // (tuỳ chọn) xóa chat UI mỗi lượt
    if (CLEAR_CHAT_EACH_TURN) {
      setMessages([]);
    }

    setMessages((prev) => [...prev, { from: "user", text }]);
    setInput("");

    const reqSeq = ++reqSeqRef.current;
    const reqId = `${Date.now()}-${reqSeq}`;

    setLoading(true);
    loadingRef.current = true;

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Request-Id": reqId },
        body: JSON.stringify({
          user_id: userId,
          thread_id: threadId,
          message: text,
        }),
      });

      let data = null;
      try {
        data = await res.json();
      } catch {
        data = null;
      }

      console.log("[chat] response", reqId, res.status, data);

      // stale response check
      if (reqSeq !== reqSeqRef.current) {
        console.warn("[chat] stale response ignored", reqId);
        return;
      }

      if (!res.ok) {
        phaseRef.current = "idle";
        setMessages((prev) => [
          ...prev,
          { from: "coach", text: "Lỗi gửi tin. Vui lòng thử lại." },
        ]);
        // cho user thử nói lại
        if (!lockedRef.current && autoListenRef.current) startListening();
        return;
      }

      const assistantText = (data && data.assistant_message) || "";
      const assistantSsml = (data && data.assistant_tts_ssml) || null;
      if (!assistantText.trim()) {
        phaseRef.current = "idle";
        setMessages((prev) => [
          ...prev,
          { from: "coach", text: "Hệ thống chưa trả về nội dung. Vui lòng thử lại." },
        ]);
        if (!lockedRef.current && autoListenRef.current) startListening();
        return;
      }

      setMessages((prev) => [...prev, { from: "coach", text: assistantText }]);

      // assistant nói xong -> auto startListening trong speakText.onend
      const spoken = speakText(assistantText, assistantSsml);
      if (!spoken) {
        // không TTS được thì quay lại nghe luôn
        phaseRef.current = "idle";
        if (!lockedRef.current && autoListenRef.current) startListening();
      }
    } catch (err) {
      console.error(err);
      phaseRef.current = "idle";
      setMessages((prev) => [
        ...prev,
        { from: "coach", text: "Không kết nối được server." },
      ]);
      if (!lockedRef.current && autoListenRef.current) startListening();
    } finally {
      setLoading(false);
      loadingRef.current = false;

      // nếu không speaking thì idle
      if (!isAgentSpeakingRef.current && !lockedRef.current) {
        if (phaseRef.current === "sending") phaseRef.current = "idle";
      }
    }
  };

  // Keep refs synced
  useEffect(() => {
    loadingRef.current = loading;
  }, [loading]);

  useEffect(() => {
    lockedRef.current = locked;
  }, [locked]);

  useEffect(() => {
    listeningRef.current = listening;
  }, [listening]);

  useEffect(() => {
    sendMessageRef.current = sendMessage;
  });

  // Scroll chat
  useEffect(() => {
    if (!listRef.current) return;
    listRef.current.scrollTop = listRef.current.scrollHeight;
  }, [messages, loading]);

  // Setup SpeechRecognition
  useEffect(() => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return;

    const recog = new SpeechRecognition();
    recog.lang = "en-US";
    recog.interimResults = true;
    recog.continuous = true;
    recog.maxAlternatives = 1;

    recog.onresult = (event) => onResultRef.current?.(event);
    recog.onend = () => onEndRef.current?.();
    recog.onerror = () => onErrorRef.current?.();

    recogRef.current = recog;

    return () => {
      try {
        recog.stop();
      } catch {
        // ignore
      }
    };
  }, []);

  // Handlers for recognition
  useEffect(() => {
    onResultRef.current = (event) => {
      let interim = "";
      let final = "";
      const offset = resultOffsetRef.current;

      for (let i = offset; i < event.results.length; i += 1) {
        const chunk = event.results[i][0].transcript;
        if (event.results[i].isFinal) final += chunk;
        else interim += chunk;
      }

      const combined = `${final}${interim}`;
      transcriptRef.current = combined;
      if (final) finalTranscriptRef.current = final;
      if (combined.trim()) heardSpeechRef.current = true;

      // track total result count for offset reset
      if (recogRef.current) recogRef.current._resultCount = event.results.length;

      // show realtime transcription
      setInput(combined);

      // nếu user đã nói và im lặng 5s -> tự gửi
      if (combined.trim()) {
        clearSilenceSendTimer();
        silenceSendTimerRef.current = setTimeout(() => {
          if (phaseRef.current !== "listening") return;
          if (loadingRef.current || lockedRef.current || isAgentSpeakingRef.current) return;
          const text = (finalTranscriptRef.current || transcriptRef.current || "").trim();
          if (!text) return;
          phaseRef.current = "sending";
          sendMessageRef.current?.(text);
        }, 5000);
      }
    };

    onEndRef.current = () => {
      setListening(false);
      listeningRef.current = false;

      // nếu đang locked / sending / speaking thì thôi
      if (lockedRef.current) return;
      if (loadingRef.current) return;
      if (isAgentSpeakingRef.current) return;

      // chỉ xử lý nếu vừa ở trạng thái listening
      if (phaseRef.current !== "listening") {
        phaseRef.current = "idle";
        return;
      }

      const text = (finalTranscriptRef.current || transcriptRef.current || "").trim();

      // clear buffers
      finalTranscriptRef.current = "";
      transcriptRef.current = "";
      setInput("");
      clearSilenceSendTimer();

      // user không nói gì -> giữ loop nghe lại để hỗ trợ realtime
      if (!text) {
        phaseRef.current = "idle";
        if (autoListenRef.current && !loadingRef.current && !lockedRef.current) {
          clearEmptySpeechRetry();
          const retryDelay = heardSpeechRef.current ? 150 : 50;
          emptySpeechRetryRef.current = setTimeout(() => {
            if (phaseRef.current === "idle") startListening();
          }, retryDelay);
        }
        return;
      }

      // user nói xong -> gửi -> agent nói
      phaseRef.current = "sending";
      sendMessageRef.current?.(text);
    };

    onErrorRef.current = () => {
      setListening(false);
      listeningRef.current = false;

      // tránh loop khi đang speaking/locked
      if (lockedRef.current) return;
      if (isAgentSpeakingRef.current) return;

      // nếu đang listening mà lỗi -> về idle rồi bật lại mic (nhẹ nhàng)
      if (phaseRef.current === "listening") {
        phaseRef.current = "idle";
      }

      // thử bật lại mic nếu cho phép
      if (autoListenRef.current && !loadingRef.current) {
        startListening();
      }
    };
  });

  // Daily check + greeting
  useEffect(() => {
    const userId = localStorage.getItem("user_id");
    const threadId = localStorage.getItem("thread_id");

    const checkDaily = async () => {
      try {
        if (!userId) {
          navigate("/login");
          return;
        }

        const res = await fetch(
          `${API_URL}/daily-status?user_id=${encodeURIComponent(userId)}`
        );

        if (!res.ok) return;

        const data = await res.json();

        // Allow users to continue learning next day immediately
        // (daily lock removed — progression is handled by current_day)

        if (!greetedRef.current) {
          greetedRef.current = true;
          const greeting =
            "Xin chào bạn, đây là chatbot hỗ trợ học tiếng Anh. Bạn vui lòng nói tiếng Anh trong suốt quá trình nhé.";

          setMessages((prev) => [...prev, { from: "coach", text: greeting }]);
          const spoken = speakText(greeting);

          // nếu không có TTS thì vẫn bật mic
          if (!spoken && autoListenRef.current) {
            phaseRef.current = "idle";
            startListening();
          }
        }

        if (!threadId) {
          navigate("/menu");
        }
      } catch {
        // fallback: vẫn greet + bật mic
        if (!greetedRef.current) {
          greetedRef.current = true;
          const greeting =
            "Xin chào bạn, đây là chatbot hỗ trợ học tiếng Anh. Bạn vui lòng nói tiếng Anh trong suốt quá trình nhé.";

          setMessages((prev) => [...prev, { from: "coach", text: greeting }]);
          const spoken = speakText(greeting);

          if (!spoken && autoListenRef.current) {
            phaseRef.current = "idle";
            startListening();
          }
        }
      }
    };

    checkDaily();
  }, [navigate]);

  const toggleMic = () => {
    if (lockedRef.current) return;

    if (listeningRef.current) {
      stopListening();
      return;
    }

    // manual bật mic -> chỉ cho bật khi idle
    if (phaseRef.current !== "idle") return;
    startListening();
  };

  return (
    <div className="min-h-screen bg-ink text-fog">
      <div className="mx-auto grid min-h-screen max-w-6xl gap-6 px-6 py-10 lg:grid-cols-[0.35fr_0.65fr]">
        <aside className="glass rounded-3xl p-6 shadow-soft">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Phiên học</h2>
            <Link to="/menu" className="text-xs uppercase tracking-[0.3em] text-haze">
              Quay lại
            </Link>
          </div>

          <div className="mt-6 space-y-4 text-sm text-haze">
            <div>
              <p className="text-base text-fog">Phiên học hôm nay</p>
            </div>

            <div>
              <p className="text-xs uppercase tracking-[0.3em] text-haze">Mục tiêu</p>
              <p className="mt-2 text-base text-fog">Nghe + nói 10 phút</p>
            </div>

            <div className="rounded-2xl border border-white/10 bg-white/5 p-4 text-xs">
              Gợi ý: Nhấn micro để luyện nói (tính năng demo).
            </div>
          </div>

          <div className="mt-6 space-y-3">
            <button className="w-full rounded-2xl bg-pine px-4 py-3 text-sm font-medium text-white">
              Tải transcript
            </button>
          </div>
        </aside>

        <section className="flex min-h-0 flex-col gap-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.3em] text-haze">Conversation</p>
              <h1 className="text-2xl font-semibold md:text-3xl">Phòng hội thoại</h1>
            </div>

            <div className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs text-haze">
              Trạng thái: {locked ? "khoá" : loading ? "đang gửi" : listening ? "đang nghe" : "đang luyện"}
            </div>
          </div>

          <div className="h-[58vh] rounded-3xl border border-white/10 bg-gradient-to-b from-white/5 to-transparent p-6 shadow-soft">
            <div className="chat-scroll h-full overflow-y-auto pr-1" ref={listRef}>
              <div className="space-y-4 pb-2">
                {messages.length === 0 ? (
                  <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-haze">
                    Nhập câu chào để bắt đầu hội thoại.
                  </div>
                ) : null}

                {messages.map((msg, index) => (
                  <div
                    key={index}
                    className={`flex ${msg.from === "user" ? "justify-end" : "justify-start"}`}
                  >
                    <div
                      className={`max-w-[75%] rounded-2xl px-4 py-3 text-sm ${msg.from === "user" ? "bg-tide text-white" : "bg-white/10 text-fog"
                        }`}
                    >
                      {msg.text}
                    </div>
                  </div>
                ))}

                {loading ? (
                  <div className="flex justify-start">
                    <div className="rounded-2xl bg-white/10 px-4 py-3 text-sm text-fog">
                      <span className="wave-dots">
                        <span />
                        <span />
                        <span />
                      </span>
                    </div>
                  </div>
                ) : null}
              </div>
            </div>
          </div>

          <div className="glass rounded-3xl p-4 shadow-soft">
            <div className="flex flex-col gap-3 md:flex-row md:items-center">
              <input
                type="text"
                placeholder="Nhập câu trả lời hoặc bấm micro..."
                value={input}
                onChange={(e) => {
                  const val = e.target.value;
                  setInput(val);
                  // If user clears input while listening, skip all old results
                  // so deleted text doesn't reappear on next speech
                  if (listeningRef.current && !val.trim()) {
                    transcriptRef.current = "";
                    finalTranscriptRef.current = "";
                    heardSpeechRef.current = false;
                    clearSilenceSendTimer();
                    // Set offset to skip all accumulated results — mic stays on
                    const count = recogRef.current?._resultCount || 0;
                    resultOffsetRef.current = count;
                  }
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") sendMessage();
                }}
                disabled={locked}
                className="flex-1 rounded-2xl border border-white/10 bg-black/30 px-4 py-3 text-sm text-fog outline-none focus:border-tide"
              />

              <div className="flex gap-3">
                <button
                  onClick={toggleMic}
                  disabled={locked}
                  className={`rounded-2xl px-4 py-3 text-sm ${listening
                    ? "border border-ember/60 bg-ember/20 text-ember"
                    : "border border-white/10 text-fog"
                    }`}
                >
                  {listening ? "Đang nghe..." : "Micro"}
                </button>

                <button
                  onClick={() => sendMessage()}
                  disabled={loading || locked}
                  className="rounded-2xl bg-ember px-4 py-3 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {loading ? "Đang gửi..." : "Gửi"}
                </button>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
