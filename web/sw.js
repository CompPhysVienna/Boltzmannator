/* Boltzmannator service worker: network-first with cache fallback.
   Fresh assets are served and cached whenever the network is available;
   the cached copy keeps the app fully usable offline. */

const CACHE = "boltzmannator-v1";

self.addEventListener("install", () => {
    self.skipWaiting();
});

self.addEventListener("activate", (e) => {
    e.waitUntil(
        caches.keys()
            .then((keys) => Promise.all(
                keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
            .then(() => self.clients.claim()));
});

self.addEventListener("fetch", (e) => {
    if (e.request.method !== "GET") return;
    e.respondWith(
        fetch(e.request).then((resp) => {
            const sameOrigin =
                new URL(e.request.url).origin === self.location.origin;
            if (resp.ok && sameOrigin) {
                const clone = resp.clone();
                caches.open(CACHE).then((c) => c.put(e.request, clone));
            }
            return resp;
        }).catch(() =>
            caches.match(e.request).then((m) => m || caches.match("./")))
    );
});
