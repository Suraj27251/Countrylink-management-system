(function () {
  'use strict';
  if (window.whatsappUtils) return;

  const parseJsonResponse = async (response, contextLabel, logPrefix = '[API]') => {
    const contentType = (response.headers.get('content-type') || '').toLowerCase();
    if (!contentType.includes('application/json')) {
      const rawText = await response.text();
      console.error(`${logPrefix} ${contextLabel} expected JSON but received ${contentType || 'unknown content-type'}.`, {
        status: response.status,
        body: rawText.slice(0, 1000)
      });
      throw new Error(`${contextLabel}: non-JSON response (HTTP ${response.status})`);
    }

    const rawText = await response.text();
    try { return rawText ? JSON.parse(rawText) : {}; }
    catch (err) {
      console.error(`${logPrefix} ${contextLabel} JSON parse failed.`, {
        status: response.status,
        contentType,
        body: rawText.slice(0, 1000),
        error: err?.message || err
      });
      throw err;
    }
  };

  const fetchWithTimeout = async (url, options = {}, timeoutMs = 20000, contextLabel = 'Fetch') => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(new Error(`Timeout after ${timeoutMs}ms`)), timeoutMs);
    try {
      const merged = { ...options, signal: controller.signal };
      return await fetch(url, merged);
    } catch (err) {
      if (err?.name === 'AbortError') {
        console.warn(`[API_TIMEOUT] ${contextLabel} timed out after ${timeoutMs}ms`);
      }
      throw err;
    } finally {
      clearTimeout(timeoutId);
    }
  };

  window.whatsappUtils = { parseJsonResponse, fetchWithTimeout };
})();
