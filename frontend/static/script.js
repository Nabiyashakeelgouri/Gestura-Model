window.Gestura = (() => {
    const API_BASE_KEY = "gestura_api_base";
    const TOKEN_KEY = "gestura_auth_token";

    function apiBase() {
        return localStorage.getItem(API_BASE_KEY) || "http://127.0.0.1:8000";
    }

    function setApiBase(value) {
        if (!value) return;
        localStorage.setItem(API_BASE_KEY, value.replace(/\/+$/, ""));
    }

    function authToken() {
        return localStorage.getItem(TOKEN_KEY) || "";
    }

    function setAuthToken(token) {
        localStorage.setItem(TOKEN_KEY, token);
    }

    function clearAuthToken() {
        localStorage.removeItem(TOKEN_KEY);
    }

    function authHeaders(extra = {}) {
        const token = authToken();
        return {
            ...extra,
            ...(token ? { Authorization: `Bearer ${token}` } : {})
        };
    }

    return {
        apiBase,
        setApiBase,
        authToken,
        setAuthToken,
        clearAuthToken,
        authHeaders
    };
})();
