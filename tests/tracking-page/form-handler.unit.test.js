/**
 * @jest-environment jsdom
 */

/**
 * Unit tests for FormHandler module edge cases.
 * Validates: Requirements 5.5, 5.8, 5.9, 9.7
 */

// Extract a standalone FormHandler for testing (mirrors the implementation in track.html)
function createFormHandler() {
    return {
        form: null,
        input: null,
        button: null,
        messagesDiv: null,
        originalButtonText: 'Track Status',
        isLoading: false,

        config: {
            maxLength: 30,
            validPattern: /^[a-zA-Z0-9-]{1,30}$/,
            timeout: 15000,
            apiEndpoint: '/api/track'
        },

        init: function(formElement) {
            this.form = formElement || document.getElementById('tracking-form');
            if (!this.form) return;

            this.input = this.form.querySelector('#tracking-input');
            this.button = this.form.querySelector('.btn-track');
            this.messagesDiv = document.getElementById('form-messages');

            if (!this.input || !this.button) return;

            this.originalButtonText = this.button.textContent;

            var self = this;
            this.form.addEventListener('submit', function(e) {
                e.preventDefault();
                self.clearMessages();
                var trackingId = self.input.value.trim();
                var result = self.validate(trackingId);
                if (!result.valid) {
                    self.showMessage(result.message, 'error');
                    self.input.focus();
                    return;
                }
                self.submit(trackingId);
            });
        },

        validate: function(trackingId) {
            if (!trackingId || trackingId.length === 0) {
                return { valid: false, message: 'Please enter a Tracking ID' };
            }
            if (!this.config.validPattern.test(trackingId)) {
                return { valid: false, message: 'Tracking ID can only contain letters, numbers, and hyphens' };
            }
            return { valid: true };
        },

        submit: function(trackingId) {
            var self = this;
            self.showLoading();
            self.clearMessages();

            var controller = new AbortController();
            var timeoutId = setTimeout(function() {
                controller.abort();
            }, self.config.timeout);

            var url = self.config.apiEndpoint + '?id=' + encodeURIComponent(trackingId);

            return fetch(url, {
                method: 'GET',
                signal: controller.signal,
                headers: { 'Accept': 'application/json' }
            })
            .then(function(response) {
                clearTimeout(timeoutId);
                if (!response.ok) {
                    if (response.status >= 400 && response.status < 500) {
                        throw new Error('Tracking ID not found. Please check and try again.');
                    } else if (response.status >= 500) {
                        throw new Error('Server error. Please try again later.');
                    }
                    throw new Error('Request failed with status ' + response.status);
                }
                return response.json().catch(function() {
                    throw new Error('Invalid response from server. Please try again.');
                });
            })
            .then(function(data) {
                self.hideLoading();
                if (data && data.status) {
                    self.showMessage('Status: ' + data.status + (data.message ? ' — ' + data.message : ''), 'success');
                } else if (data && data.message) {
                    self.showMessage(data.message, 'info');
                } else {
                    self.showMessage('Tracking information retrieved successfully.', 'success');
                }
            })
            .catch(function(error) {
                clearTimeout(timeoutId);
                self.hideLoading();
                if (error.name === 'AbortError') {
                    self.showMessage('Request timed out. Please try again.', 'error');
                } else if (error.message === 'Failed to fetch' || error.name === 'TypeError') {
                    self.showMessage('Network error. Please check your connection and try again.', 'error');
                } else {
                    self.showMessage(error.message || 'An unexpected error occurred. Please try again.', 'error');
                }
            });
        },

        showLoading: function() {
            if (!this.button) return;
            this.isLoading = true;
            this.button.disabled = true;
            this.button.innerHTML = '<span class="spinner"></span> Loading...';
        },

        hideLoading: function() {
            if (!this.button) return;
            this.isLoading = false;
            this.button.disabled = false;
            this.button.textContent = this.originalButtonText;
        },

        showMessage: function(message, type) {
            if (!this.messagesDiv) return;
            this.messagesDiv.textContent = message;
            this.messagesDiv.className = 'form-messages ' + (type || 'info');
        },

        clearMessages: function() {
            if (!this.messagesDiv) return;
            this.messagesDiv.textContent = '';
            this.messagesDiv.className = 'form-messages';
        }
    };
}

/**
 * Set up a minimal DOM structure matching track.html's form.
 */
function setupDOM() {
    document.body.innerHTML = `
        <form id="tracking-form" novalidate>
            <label for="tracking-input" class="sr-only">Tracking ID</label>
            <input type="text" id="tracking-input" class="input-track" placeholder="Enter Tracking ID" maxlength="30" autocomplete="off" aria-label="Tracking ID">
            <button type="submit" class="btn-track">Track Status</button>
            <div id="form-messages" class="form-messages" aria-live="assertive" role="status"></div>
        </form>
    `;
}

describe('FormHandler - Validation Edge Cases', () => {
    let handler;

    beforeEach(() => {
        setupDOM();
        handler = createFormHandler();
        handler.init();
    });

    // Validates: Requirement 5.5
    test('empty input returns invalid with required message', () => {
        const result = handler.validate('');
        expect(result.valid).toBe(false);
        expect(result.message).toBe('Please enter a Tracking ID');
    });

    test('null input returns invalid with required message', () => {
        const result = handler.validate(null);
        expect(result.valid).toBe(false);
        expect(result.message).toBe('Please enter a Tracking ID');
    });

    test('undefined input returns invalid with required message', () => {
        const result = handler.validate(undefined);
        expect(result.valid).toBe(false);
        expect(result.message).toBe('Please enter a Tracking ID');
    });

    // Boundary length tests
    test('single character "a" is valid', () => {
        const result = handler.validate('a');
        expect(result.valid).toBe(true);
        expect(result.message).toBeUndefined();
    });

    test('30 characters is valid (upper boundary)', () => {
        const input = 'abcdefghijklmnopqrstuvwxyz1234'; // exactly 30 chars
        expect(input.length).toBe(30);
        const result = handler.validate(input);
        expect(result.valid).toBe(true);
    });

    test('31 characters is invalid (exceeds boundary)', () => {
        const input = 'abcdefghijklmnopqrstuvwxyz12345'; // 31 chars
        expect(input.length).toBe(31);
        const result = handler.validate(input);
        expect(result.valid).toBe(false);
        expect(result.message).toBe('Tracking ID can only contain letters, numbers, and hyphens');
    });

    // Special character tests
    test('special characters "abc@def" are rejected', () => {
        const result = handler.validate('abc@def');
        expect(result.valid).toBe(false);
        expect(result.message).toBe('Tracking ID can only contain letters, numbers, and hyphens');
    });

    test('valid input with hyphens "abc-123-def" is accepted', () => {
        const result = handler.validate('abc-123-def');
        expect(result.valid).toBe(true);
    });

    test('spaces are rejected', () => {
        const result = handler.validate('abc def');
        expect(result.valid).toBe(false);
        expect(result.message).toBe('Tracking ID can only contain letters, numbers, and hyphens');
    });

    test('underscores are rejected', () => {
        const result = handler.validate('abc_def');
        expect(result.valid).toBe(false);
    });
});

describe('FormHandler - Loading State Toggle', () => {
    let handler;

    beforeEach(() => {
        setupDOM();
        handler = createFormHandler();
        handler.init();
    });

    // Validates: Requirement 5.8
    test('showLoading disables the button', () => {
        handler.showLoading();
        expect(handler.button.disabled).toBe(true);
    });

    test('showLoading changes button text to loading indicator', () => {
        handler.showLoading();
        expect(handler.button.textContent).toContain('Loading');
    });

    test('showLoading sets isLoading flag to true', () => {
        handler.showLoading();
        expect(handler.isLoading).toBe(true);
    });

    test('hideLoading re-enables the button', () => {
        handler.showLoading();
        handler.hideLoading();
        expect(handler.button.disabled).toBe(false);
    });

    test('hideLoading restores original button text', () => {
        handler.showLoading();
        handler.hideLoading();
        expect(handler.button.textContent).toBe('Track Status');
    });

    test('hideLoading sets isLoading flag to false', () => {
        handler.showLoading();
        handler.hideLoading();
        expect(handler.isLoading).toBe(false);
    });
});

describe('FormHandler - Error Display in ARIA Live Region', () => {
    let handler;

    beforeEach(() => {
        setupDOM();
        handler = createFormHandler();
        handler.init();
    });

    // Validates: Requirement 9.7
    test('showMessage sets textContent on messages div', () => {
        handler.showMessage('Test error message', 'error');
        expect(handler.messagesDiv.textContent).toBe('Test error message');
    });

    test('showMessage applies error class', () => {
        handler.showMessage('Error occurred', 'error');
        expect(handler.messagesDiv.className).toBe('form-messages error');
    });

    test('showMessage applies success class', () => {
        handler.showMessage('Success!', 'success');
        expect(handler.messagesDiv.className).toBe('form-messages success');
    });

    test('showMessage applies info class', () => {
        handler.showMessage('Info message', 'info');
        expect(handler.messagesDiv.className).toBe('form-messages info');
    });

    test('messages div has aria-live="assertive" attribute', () => {
        const messagesDiv = document.getElementById('form-messages');
        expect(messagesDiv.getAttribute('aria-live')).toBe('assertive');
    });

    test('messages div has role="status" attribute', () => {
        const messagesDiv = document.getElementById('form-messages');
        expect(messagesDiv.getAttribute('role')).toBe('status');
    });

    test('clearMessages removes text content', () => {
        handler.showMessage('Some message', 'error');
        handler.clearMessages();
        expect(handler.messagesDiv.textContent).toBe('');
    });

    test('clearMessages resets class to base', () => {
        handler.showMessage('Some message', 'error');
        handler.clearMessages();
        expect(handler.messagesDiv.className).toBe('form-messages');
    });
});

describe('FormHandler - Timeout Handling', () => {
    let handler;

    beforeEach(() => {
        setupDOM();
        handler = createFormHandler();
        handler.init();
        jest.useFakeTimers();
    });

    afterEach(() => {
        jest.useRealTimers();
        jest.restoreAllMocks();
    });

    // Validates: Requirement 5.9
    test('request aborts after 15 seconds and shows timeout error', async () => {
        // Mock fetch to return a promise that never resolves (simulating a hanging request)
        const abortError = new DOMException('The operation was aborted.', 'AbortError');
        let fetchSignal;

        global.fetch = jest.fn(function(url, options) {
            fetchSignal = options.signal;
            return new Promise(function(resolve, reject) {
                // Listen for abort signal
                if (options.signal) {
                    options.signal.addEventListener('abort', function() {
                        reject(abortError);
                    });
                }
            });
        });

        const submitPromise = handler.submit('valid-id');

        // Advance time by 15 seconds to trigger the timeout
        jest.advanceTimersByTime(15000);

        await submitPromise;

        // Verify the timeout error message is displayed
        expect(handler.messagesDiv.textContent).toBe('Request timed out. Please try again.');
        expect(handler.messagesDiv.className).toContain('error');
        // Verify button is re-enabled after timeout
        expect(handler.button.disabled).toBe(false);
        expect(handler.isLoading).toBe(false);
    });

    test('timeout is configured to 15000ms', () => {
        expect(handler.config.timeout).toBe(15000);
    });
});
