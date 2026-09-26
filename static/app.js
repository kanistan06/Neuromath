/* App state */
let currentUser = null;   // { student_id, name }
let quizPaper   = [];
let practicePaper = [];
let currentPage = 'dashboard';
let quizLoadInFlight = false;
let practiceTabs = [];
let activePracticeTabId = '';
let selectedPracticeGrade = null;
let selectedPracticeTopic = '';
let csrfToken = null;
let pendingResetEmail = '';
let diagnosticQuizId = '';
let practiceQuizId = '';
let diagnosticBatchIndex = 0;
let diagnosticTotalCount = 0;
let diagnosticHasMore = false;
let practiceBatchIndex = 0;
let practiceTotalCount = 0;
let practiceHasMore = false;
const PRACTICE_BATCH_SIZE = 5;
const DIAGNOSTIC_BATCH_SIZE = 5;
let diagnosticTimerState = { enabled: false, intervalId: null, deadlineMs: 0, autoSubmitting: false };
let practiceTimerState = { enabled: false, intervalId: null, deadlineMs: 0, autoSubmitting: false };
let brainGraphLayoutCache = null;
let brainGraphLayoutVersion = 0;
const BRAIN_LAYOUT_VERSION = 6;
let menuPreloadPromise = null;
const preloadedMenuData = {
  attempts: null,
  guidance: null,
  practice: null,
  settings: null,
  profile: null,
};

/** Closed polygon approximating brainOutlinePathD (extra vertices on curved segments). */
const BRAIN_SILHOUETTE_POLY = [
  [24, 56], [17, 38], [15, 18], [30, 8], [50, 3.5], [72, 6], [88, 18],
  [93, 36], [90, 54], [84, 64], [74, 71], [82, 76], [87, 82],
  [78, 88], [62, 90], [52, 89], [40, 80], [32, 70], [26, 62],
];

function pointInPolygon(x, y, poly) {
  let inside = false;
  const n = poly.length;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = poly[i][0];
    const yi = poly[i][1];
    const xj = poly[j][0];
    const yj = poly[j][1];
    const dy = yj - yi;
    if (Math.abs(dy) < 1e-9) continue;
    const intersect = (yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / dy + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}
const GAMIFICATION_BADGE_FALLBACK_TOTAL = 13;

/* SVG defs: score ring gradient */
(function injectSvgDefs() {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('width', '0'); svg.setAttribute('height', '0');
  svg.style.position = 'absolute';
  svg.innerHTML = `<defs>
    <linearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%"   stop-color="#3ecf8e"/>
      <stop offset="100%" stop-color="#5b9cf5"/>
    </linearGradient>
  </defs>`;
  document.body.prepend(svg);
})();

/* Helpers */
function esc(val) {
  const s = document.createElement('span');
  s.textContent = String(val ?? '');
  return s.innerHTML;
}

/** MCQ diagram from server (/static/quiz_images/<uuid>.png only). */
function questionFigureHtml(q) {
  const url = q && typeof q.image_url === 'string' ? q.image_url.trim() : '';
  if (!/^\/static\/quiz_images\/[a-f0-9]{32}\.png$/i.test(url)) return '';
  return `<figure class="question-figure"><img src="${url}" alt="Question diagram" loading="lazy" decoding="async" /></figure>`;
}

function setBtn(id, loading) {
  const btn    = document.getElementById(id);
  const label  = document.getElementById(id + 'Label');
  const spin   = document.getElementById('spin' + id.replace('btn', '').replace(/^./, c => c.toUpperCase()));
  if (!btn) return;
  btn.disabled = loading;
  if (label) label.style.opacity = loading ? '0' : '1';
  if (spin)  spin.classList.toggle('hidden', !loading);
}

function showError(id, msg) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = msg;
  el.classList.remove('success');
  el.classList.remove('hidden');
}

function showAuthMessage(id, msg) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = msg;
  el.classList.add('success');
  el.classList.remove('hidden');
}

function hideError(id) {
  const el = document.getElementById(id);
  if (el) el.classList.add('hidden');
}

function quizUserMessage(data, fallback) {
  const code = String(data?.code || '');
  const messages = {
    quiz_preparing: 'Your quiz is already being prepared. Please wait a moment and try again.',
    next_batch_preparing: 'The next five questions are still being prepared. Please wait a moment.',
    inference_credits_exhausted: 'Question generation is temporarily unavailable. Please try again shortly.',
    generation_budget_exhausted: 'Question generation is taking longer than expected. Please try again shortly.',
    hf_embeddings_unavailable: 'The learning content is temporarily unavailable. Please try again shortly.',
    question_pool_insufficient: 'Diagnostic questions could not be prepared. Please try again shortly.',
    question_pool_fallback_failed: 'Practice questions could not be prepared. Please try again shortly.',
  };
  if (messages[code]) return messages[code];
  if (data?.status === 'preparing') {
    return 'Your questions are being prepared. Please wait a moment and try again.';
  }
  return fallback;
}

function resetMenuPreloadCache() {
  menuPreloadPromise = null;
  Object.keys(preloadedMenuData).forEach((key) => {
    preloadedMenuData[key] = null;
  });
}

async function fetchPreloadJson(url) {
  const res = await fetch(url, { credentials: 'same-origin' });
  const data = await res.json();
  return res.ok && data && data.status === 'ok' ? data : null;
}

function preloadAuthenticatedMenuData() {
  if (!currentUser || menuPreloadPromise) return menuPreloadPromise;

  const tasks = [];
  // Dashboard loading already fetches /api/attempts and /api/recommend and
  // stores those responses in the same preload cache. Load the remaining
  // menu data concurrently so navigation does not wait on first use.
  if (!preloadedMenuData.practice) {
    tasks.push(fetchPreloadJson('/api/practice/catalog').then((data) => {
      if (data) preloadedMenuData.practice = data;
    }));
  }
  if (!preloadedMenuData.settings) {
    tasks.push(fetchPreloadJson('/api/settings').then((data) => {
      if (data) preloadedMenuData.settings = data;
    }));
  }
  if (!preloadedMenuData.profile) {
    tasks.push(fetchPreloadJson('/api/profile').then((data) => {
      if (data) preloadedMenuData.profile = data;
    }));
  }

  menuPreloadPromise = Promise.allSettled(tasks).finally(() => {
    menuPreloadPromise = null;
  });
  return menuPreloadPromise;
}

function readValidEmail(inputId, errorId) {
  const input = document.getElementById(inputId);
  const email = input?.value.trim() || '';
  if (!email) {
    showError(errorId, 'Email is required.');
    input?.focus();
    return null;
  }
  if (!input.checkValidity()) {
    showError(errorId, 'Enter a valid email address.');
    input.focus();
    return null;
  }
  return email;
}

function setPracticeLoading(loading, message = 'Loading questions...') {
  const overlay = document.getElementById('practiceLoadingOverlay');
  const textEl = document.getElementById('practiceLoadingText');
  if (!overlay) return;
  if (textEl) textEl.textContent = message;
  overlay.classList.toggle('hidden', !loading);
}

function flashOk(id, msg = 'Saved') {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = msg;
  el.classList.remove('hidden');
  setTimeout(() => el.classList.add('hidden'), 2500);
}

function timerState(kind) {
  return kind === 'practice' ? practiceTimerState : diagnosticTimerState;
}

function timerElements(kind) {
  return kind === 'practice'
    ? { wrap: document.getElementById('practiceQuizTimer'), value: document.getElementById('practiceQuizTimerValue') }
    : { wrap: document.getElementById('quizTimer'), value: document.getElementById('quizTimerValue') };
}

function stopQuizTimer(kind) {
  const state = timerState(kind);
  if (state.intervalId) clearInterval(state.intervalId);
  state.intervalId = null;
  state.deadlineMs = 0;
  state.autoSubmitting = false;
  state.enabled = false;
  const { wrap, value } = timerElements(kind);
  if (wrap) {
    wrap.classList.add('hidden');
    wrap.classList.remove('warning', 'danger');
  }
  if (value) value.textContent = '00:00';
}

function formatQuizTime(totalSeconds) {
  const safe = Math.max(0, Math.ceil(Number(totalSeconds) || 0));
  const minutes = Math.floor(safe / 60);
  const seconds = safe % 60;
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

function lockQuizInputs(prefix) {
  document.querySelectorAll(`input[name^="${prefix}"]`).forEach(input => { input.disabled = true; });
}

function startQuizTimer(kind, timer) {
  stopQuizTimer(kind);
  if (!timer || timer.enabled !== true) return;

  const state = timerState(kind);
  const { wrap, value } = timerElements(kind);
  if (!wrap || !value) return;

  const remaining = Math.max(0, Number(timer.remaining_seconds ?? timer.duration_seconds ?? 0));
  state.enabled = true;
  state.deadlineMs = Date.now() + remaining * 1000;
  wrap.classList.remove('hidden');

  const tick = () => {
    const secondsLeft = Math.max(0, Math.ceil((state.deadlineMs - Date.now()) / 1000));
    value.textContent = formatQuizTime(secondsLeft);
    wrap.classList.toggle('warning', secondsLeft > 60 && secondsLeft <= 300);
    wrap.classList.toggle('danger', secondsLeft <= 60);

    if (secondsLeft <= 0 && !state.autoSubmitting) {
      state.autoSubmitting = true;
      state.enabled = false;
      if (state.intervalId) clearInterval(state.intervalId);
      state.intervalId = null;
      value.textContent = '00:00';
      lockQuizInputs(kind === 'practice' ? 'pq' : 'q');
      if (kind === 'practice') {
        showError('practiceSubmitValidation', "Time is up. Your practice quiz is being submitted automatically.");
        submitPracticeQuiz(null, true);
      } else {
        showError('quizSubmitValidation', "Time is up. Your quiz is being submitted automatically.");
        submitQuiz(null, true);
      }
    }
  };

  tick();
  if (!state.autoSubmitting) state.intervalId = setInterval(tick, 1000);
}

function unansweredQuestionIndices(paper, prefix) {
  const missing = [];
  paper.forEach((_, index) => {
    if (!document.querySelector(`input[name="${prefix}${index}"]:checked`)) missing.push(index);
  });
  return missing;
}

function focusUnansweredQuestion(kind, index, unansweredCount) {
  const practice = kind === 'practice';
  if (practice) {
    practiceBatchIndex = Math.floor(Number(index || 0) / PRACTICE_BATCH_SIZE);
    renderPracticeBatch();
  }
  const card = document.getElementById(`${practice ? 'practice-question' : 'quiz-question'}-${index}`);
  const errorId = practice ? 'practiceSubmitValidation' : 'quizSubmitValidation';
  const prefix = practice ? 'pq' : 'q';
  const suffix = unansweredCount > 1 ? ` ${unansweredCount} questions are unanswered.` : '';
  showError(errorId, `Please select an answer for Question ${index + 1} before submitting.${suffix}`);
  if (card) {
    card.classList.add('question-needs-answer');
    card.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
  document.querySelector(`input[name="${prefix}${index}"]`)?.focus({ preventScroll: true });
}

/* Auth */
async function ensureCsrfToken() {
  if (csrfToken) return csrfToken;
  const res = await fetch('/api/csrf-token', { credentials: 'same-origin' });
  const data = await res.json();
  if (!res.ok || !data.csrf_token) throw new Error('Could not initialize the secure session.');
  csrfToken = data.csrf_token;
  return csrfToken;
}

async function apiFetch(url, options = {}) {
  const method = String(options.method || 'GET').toUpperCase();
  const headers = new Headers(options.headers || {});
  if (!['GET', 'HEAD', 'OPTIONS'].includes(method)) {
    headers.set('X-CSRFToken', await ensureCsrfToken());
  }
  return fetch(url, { ...options, method, headers, credentials: 'same-origin' });
}

function switchAuthView(view) {
  ['signin', 'signup', 'forgot', 'reset'].forEach(t => {
    document.getElementById(`panel-${t}`)?.classList.toggle('active', t === view);
  });
  ['signin', 'signup'].forEach(t => {
    document.getElementById(`tab-${t}-btn`).classList.toggle('active', t === view);
    document.getElementById(`tab-${t}-btn`).setAttribute('aria-selected', t === view);
  });
  ['siError', 'suError', 'forgotError', 'resetError'].forEach(hideError);
  document.getElementById('btnResendVerification')?.classList.add('hidden');
}

function switchAuthTab(tab) {
  switchAuthView(tab);
}

async function doSignIn() {
  hideError('siError');
  const email    = readValidEmail('siEmail', 'siError');
  const password = document.getElementById('siPassword').value;
  if (!email) return;
  if (!password) { showError('siError', 'Password is required.'); return; }
  setBtn('btnSignIn', true);
  try {
    const res  = await apiFetch('/api/signin', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ email, password }) });
    const data = await res.json();
    if (!res.ok || data.status !== 'ok') {
      showError('siError', data.message || 'Sign in failed.');
      document.getElementById('btnResendVerification')?.classList.toggle('hidden', !data.needs_verification);
      return;
    }
    csrfToken = null;
    onAuthenticated(data);
  } catch (e) { showError('siError', e.message); }
  finally { setBtn('btnSignIn', false); }
}

async function doSignUp() {
  hideError('suError');
  const name     = document.getElementById('suName').value.trim();
  const email    = readValidEmail('suEmail', 'suError');
  const password = document.getElementById('suPassword').value;
  const confirmation = document.getElementById('suPasswordConfirm').value;
  if (!name)     { showError('suError', 'Name is required.'); return; }
  if (!email) return;
  if (!password) { showError('suError', 'Password is required.'); return; }
  if (password !== confirmation) { showError('suError', 'Passwords do not match.'); return; }
  setBtn('btnSignUp', true);
  try {
    const res  = await apiFetch('/api/signup', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ name, email, password }) });
    const data = await res.json();
    if (!res.ok || data.status !== 'ok') { showError('suError', data.message || 'Sign up failed.'); return; }
    switchAuthTab('signin');
    document.getElementById('siEmail').value = email;
    showAuthMessage('siError', data.message || 'Check your email to verify your account.');
  } catch (e) { showError('suError', e.message); }
  finally { setBtn('btnSignUp', false); }
}

async function doSignOut() {
  await apiFetch('/api/signout', { method: 'POST' });
  csrfToken = null;
  currentUser = null;
  resetMenuPreloadCache();
  brainGraphLayoutCache = null;
  brainGraphLayoutVersion = 0;
  quizPaper   = [];
  practicePaper = [];
  diagnosticQuizId = '';
  practiceQuizId = '';
  stopQuizTimer('diagnostic');
  stopQuizTimer('practice');
  document.getElementById('app').classList.add('hidden');
  document.getElementById('auth-overlay').style.display = 'flex';
  // Clear auth fields
  document.getElementById('siEmail').value    = '';
  document.getElementById('siPassword').value = '';
  document.getElementById('suName').value     = '';
  document.getElementById('suEmail').value    = '';
  document.getElementById('suPassword').value = '';
  document.getElementById('suPasswordConfirm').value = '';
  switchAuthTab('signin');
}

async function doResendVerification() {
  hideError('siError');
  const email = readValidEmail('siEmail', 'siError');
  if (!email) return;
  try {
    const res = await apiFetch('/api/resend-verification', {
      method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ email }),
    });
    const data = await res.json();
    if (!res.ok) { showError('siError', data.message || 'Could not resend verification email.'); return; }
    showAuthMessage('siError', data.message);
  } catch (e) { showError('siError', e.message); }
}

async function doForgotPassword() {
  hideError('forgotError');
  const email = readValidEmail('forgotEmail', 'forgotError');
  if (!email) return;
  setBtn('btnForgotPassword', true);
  try {
    const res = await apiFetch('/api/forgot-password', {
      method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ email }),
    });
    const data = await res.json();
    if (!res.ok) { showError('forgotError', data.message || 'Could not request a reset OTP.'); return; }
    pendingResetEmail = email;
    document.getElementById('resetEmail').value = email;
    document.getElementById('resetOtp').value = '';
    switchAuthView('reset');
    showAuthMessage('resetError', data.message);
  } catch (e) { showError('forgotError', e.message); }
  finally { setBtn('btnForgotPassword', false); }
}

function enableAuthEnterSubmission() {
  const bindings = [
    ['panel-signin', 'btnSignIn'],
    ['panel-signup', 'btnSignUp'],
    ['panel-forgot', 'btnForgotPassword'],
  ];

  bindings.forEach(([panelId, buttonId]) => {
    const panel = document.getElementById(panelId);
    if (!panel) return;
    panel.addEventListener('keydown', (event) => {
      if (event.key !== 'Enter' || event.isComposing) return;
      if (!(event.target instanceof HTMLInputElement)) return;
      const button = document.getElementById(buttonId);
      if (!button || button.disabled) return;
      event.preventDefault();
      button.click();
    });
  });
}

async function doResetPassword() {
  hideError('resetError');
  const email = readValidEmail('resetEmail', 'resetError');
  const otp = document.getElementById('resetOtp').value.trim();
  const password = document.getElementById('resetPassword').value;
  const confirmation = document.getElementById('resetPasswordConfirm').value;
  if (!email) return;
  if (!/^\d{6}$/.test(otp)) { showError('resetError', 'Enter the 6-digit OTP from your email.'); return; }
  if (!password) { showError('resetError', 'Password is required.'); return; }
  if (password !== confirmation) { showError('resetError', 'Passwords do not match.'); return; }
  setBtn('btnResetPassword', true);
  try {
    const res = await apiFetch('/api/reset-password', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ email, otp, password }),
    });
    const data = await res.json();
    if (!res.ok) { showError('resetError', data.message || 'Could not reset the password.'); return; }
    pendingResetEmail = '';
    document.getElementById('resetOtp').value = '';
    document.getElementById('resetPassword').value = '';
    document.getElementById('resetPasswordConfirm').value = '';
    csrfToken = null;
    switchAuthTab('signin');
    showAuthMessage('siError', data.message);
  } catch (e) { showError('resetError', e.message); }
  finally { setBtn('btnResetPassword', false); }
}

async function handleAuthLink() {
  const params = new URLSearchParams(window.location.hash.replace(/^#/, ''));
  const verifyToken = params.get('verify') || '';
  if (!verifyToken) return;
  history.replaceState(null, '', `${window.location.pathname}${window.location.search}`);
  document.getElementById('auth-overlay').style.display = 'flex';
  switchAuthTab('signin');
  try {
    const res = await apiFetch('/api/verify-email', {
      method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ token: verifyToken }),
    });
    const data = await res.json();
    if (!res.ok) { showError('siError', data.message || 'Verification failed.'); return; }
    showAuthMessage('siError', data.message);
  } catch (e) { showError('siError', e.message); }
}

function onAuthenticated(data) {
  resetMenuPreloadCache();
  currentUser = { email: data.email, name: data.name || data.email };
  applyUserToUI();
  document.getElementById('auth-overlay').style.display = 'none';
  document.getElementById('app').classList.remove('hidden');
  applyStoredSidebarLayout();
  showPage('dashboard');
  void preloadAuthenticatedMenuData();
}

function applyUserToUI() {
  if (!currentUser) return;
  const initials = currentUser.name.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase();
  document.getElementById('userAvatar').textContent = initials;
  document.getElementById('userName').textContent   = currentUser.name;
}

/* Navigation */
const PAGE_TITLES = {
  dashboard : 'Dashboard',
  attempts  : 'Attempt Papers',
  quiz      : 'MCQ Quiz',
  guidance  : 'Study Guidance',
  practice  : 'Practice Quiz',
  settings  : 'Settings',
};

function isNarrowNav() {
  return window.matchMedia('(max-width: 1024px)').matches;
}

function showPage(name) {
  currentPage = name;
  document.querySelectorAll('.page').forEach(p => p.classList.toggle('active', p.id === `page-${name}`));
  document.querySelectorAll('.nav-item[data-page]').forEach(n => n.classList.toggle('active', n.dataset.page === name));
  document.getElementById('topbarTitle').textContent = PAGE_TITLES[name] || name;
  if (isNarrowNav()) {
    document.getElementById('sidebar').classList.remove('open');
    const backdrop = document.getElementById('sidebarBackdrop');
    if (backdrop) backdrop.classList.remove('open');
  }
  syncMobileMenuButton();
  syncSidebarPinUi();
  if (name === 'dashboard') loadDashboard();
  if (name === 'attempts') loadAttemptPapers();
  if (name === 'guidance') loadGuidance();
  if (name === 'practice') loadPracticeCatalog();
}

function syncMobileMenuButton() {
  const sidebar = document.getElementById('sidebar');
  const btn = document.getElementById('mobileMenuBtn');
  const icon = document.getElementById('mobileMenuIcon');
  const app = document.getElementById('app');
  if (!sidebar || !btn) return;
  const navOpen = isNarrowNav()
    ? sidebar.classList.contains('open')
    : !app?.classList.contains('sidebar-collapsed');
  btn.setAttribute('aria-expanded', navOpen ? 'true' : 'false');
  btn.setAttribute('aria-label', navOpen ? 'Close menu' : 'Open menu');
  if (icon) icon.textContent = navOpen ? 'close' : 'menu';
}

function syncSidebarPinUi() {
  const app = document.getElementById('app');
  const btn = document.getElementById('sidebarPinBtn');
  if (!btn || !app) return;
  const collapsed = app.classList.contains('sidebar-collapsed');
  btn.setAttribute('aria-pressed', collapsed ? 'true' : 'false');
  btn.setAttribute('aria-label', collapsed ? 'Pin sidebar open' : 'Hide sidebar');
  const icon = btn.querySelector('.sidebar-pin-icon');
  const label = btn.querySelector('.sidebar-pin-label');
  if (icon) icon.textContent = collapsed ? 'last_page' : 'first_page';
  if (label) label.textContent = collapsed ? 'Pin sidebar open' : 'Hide sidebar';
}

function applyStoredSidebarLayout() {
  const app = document.getElementById('app');
  if (!app) return;
  try {
    if (localStorage.getItem('synapSidebarCollapsed') === '1' && window.matchMedia('(min-width: 1025px)').matches) {
      app.classList.add('sidebar-collapsed');
    }
  } catch (_) {}
  syncSidebarPinUi();
  syncMobileMenuButton();
}

function toggleSidebarPin() {
  if (isNarrowNav()) return;
  toggleSidebar();
}

function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  const backdrop = document.getElementById('sidebarBackdrop');
  const app = document.getElementById('app');
  if (!sidebar || !app) return;

  if (isNarrowNav()) {
    sidebar.classList.toggle('open');
    const isOpen = sidebar.classList.contains('open');
    if (backdrop) backdrop.classList.toggle('open', isOpen);
  } else {
    app.classList.toggle('sidebar-collapsed');
    sidebar.classList.remove('open');
    if (backdrop) backdrop.classList.remove('open');
    try {
      localStorage.setItem('synapSidebarCollapsed', app.classList.contains('sidebar-collapsed') ? '1' : '0');
    } catch (_) {}
    syncSidebarPinUi();
  }
  syncMobileMenuButton();
}
function formatSystemTime(isoText) {
  if (!isoText) return '';
  const hasTimezone = /([zZ]|[+\-]\d{2}:?\d{2})$/.test(isoText);
  const normalized = hasTimezone ? isoText : `${isoText}Z`;
  const dt = new Date(normalized);
  if (Number.isNaN(dt.getTime())) return String(isoText);
  return dt.toLocaleString();
}

function goToAttempts() {
  showPage('attempts');
}

/* Dashboard */
async function loadDashboard() {
  if (!currentUser) return;
  document.getElementById('welcomeHeading').textContent = `Welcome back, ${currentUser.name}!`;

  // Primary dashboard source: attempts API
  let attempts = [];
  try {
    const res = await fetch('/api/attempts');
    const data = await res.json();
    if (res.ok && data.status === 'ok') {
      attempts = data.attempts || [];
      preloadedMenuData.attempts = attempts;
    } else {
      attempts = [];
    }
  } catch (_) {
    attempts = [];
  }

  const scores = attempts.map(a => Number(a.score_percent)).filter(v => Number.isFinite(v));
  const avg = scores.length ? Math.round(scores.reduce((a, b) => a + b, 0) / scores.length) : null;
  const last = scores.length ? scores[0] : null; // latest first
  document.getElementById('statAttempts').textContent = attempts.length || '0';
  document.getElementById('statAvgScore').textContent = avg != null ? `${avg}%` : '-';
  document.getElementById('statLastScore').textContent = last != null ? `${Math.round(last)}%` : '-';

  // Secondary source: recommendation endpoint for weak topics only
  try {
    const res = await fetch('/api/recommend');
    const data = await res.json();
    if (res.ok && data.status === 'ok') {
      preloadedMenuData.guidance = data;
      document.getElementById('statWeakTopics').textContent = (data.weak_topics || []).length || '0';
    } else {
      document.getElementById('statWeakTopics').textContent = '0';
    }
  } catch (_) {
    document.getElementById('statWeakTopics').textContent = '0';
  }

  try {
    const gRes = await fetch('/api/gamification');
    const gData = await gRes.json();
    if (gRes.ok && gData.status === 'ok') {
      renderSynapticBrain(gData.attempts_by_day || {}, gData.stats || {});
      renderAchievementBadges(gData.badges || [], gData.earned_count, gData.total_badges);
    } else {
      renderSynapticBrain({}, {});
      renderAchievementBadges([], 0, GAMIFICATION_BADGE_FALLBACK_TOTAL);
    }
  } catch (_) {
    renderSynapticBrain({}, {});
    renderAchievementBadges([], 0, GAMIFICATION_BADGE_FALLBACK_TOTAL);
  }
}

function hashStr(s) {
  let h = 0;
  const str = String(s ?? '');
  for (let i = 0; i < str.length; i++) h = Math.imul(31, h) + str.charCodeAt(i) | 0;
  return Math.abs(h);
}

function insideBrainSilhouette(x, y) {
  return pointInPolygon(x, y, BRAIN_SILHOUETTE_POLY);
}

function brainOutlinePathD() {
  return [
    'M 24 56',
    'C 14 44 11 28 15 18',
    'C 19 8 34 3 50 3.5',
    'C 68 4 84 12 92 26',
    'C 98 40 96 54 88 62',
    'C 82 68 76 71 70 72',
    'C 78 73 88 78 86 84',
    'C 84 89 76 90 68 87',
    'C 62 88 58 91 52 89',
    'C 46 86 42 80 36 74',
    'C 30 68 26 62 24 56',
    'Z',
  ].join(' ');
}

function brainFissurePathD() {
  return [
    'M 44 6 C 40 18 36 32 34 46',
    'M 18 44 C 32 38 50 34 72 32 C 78 32 84 36 88 42',
  ].join(' ');
}

function brainSulciDecorSvg() {
  return [
    '<path class="brain-sulcus" d="M 76 14 C 74 24 72 34 70 44" fill="none"/>',
    '<path class="brain-sulcus" d="M 84 22 C 86 30 87 40 84 50" fill="none"/>',
    '<path class="brain-sulcus" d="M 62 6 C 60 16 58 26 56 36" fill="none"/>',
    '<path class="brain-sulcus" d="M 28 12 C 34 10 40 11 44 8" fill="none"/>',
    '<path class="brain-sulcus" d="M 22 22 C 30 18 38 17 46 15" fill="none"/>',
    '<path class="brain-sulcus" d="M 20 34 C 28 30 36 28 44 26" fill="none"/>',
    '<path class="brain-sulcus" d="M 36 8 Q 40 12 38 18" fill="none"/>',
    '<path class="brain-sulcus" d="M 50 6 Q 52 14 50 22" fill="none"/>',
    '<path class="brain-sulcus" d="M 48 28 C 46 36 44 44 42 50" fill="none"/>',
    '<path class="brain-sulcus" d="M 58 18 C 56 28 54 38 52 46" fill="none"/>',
    '<path class="brain-sulcus" d="M 32 42 C 38 40 44 41 48 44" fill="none"/>',
    '<path class="brain-sulcus" d="M 26 54 C 34 52 42 53 50 56" fill="none"/>',
    '<path class="brain-sulcus" d="M 28 62 C 36 60 44 60 52 62" fill="none"/>',
    '<path class="brain-sulcus" d="M 30 70 C 38 68 46 68 54 70" fill="none"/>',
    '<path class="brain-sulcus brain-sulcus--cereb" d="M 78 74 C 82 73 86 74 89 76" fill="none"/>',
    '<path class="brain-sulcus brain-sulcus--cereb" d="M 77 78 C 81 77 85 78 88 80" fill="none"/>',
    '<path class="brain-sulcus brain-sulcus--cereb" d="M 76 82 C 80 81 84 82 86 84" fill="none"/>',
    '<path class="brain-sulcus brain-sulcus--cereb" d="M 75 86 C 78 85 81 86 83 87" fill="none"/>',
  ].join('');
}

function buildBrainGraph() {
  const salt = hashStr(currentUser?.email || 'anon');
  let seed = salt || 1;
  const rnd = () => {
    seed = (Math.imul(seed, 1103515245) + 12345) | 0;
    return (seed >>> 0) / 0xffffffff;
  };

  const nodes = [];
  for (let k = 0; k < 24000 && nodes.length < 260; k++) {
    const x = 8 + rnd() * 90;
    const y = 2 + rnd() * 92;
    if (insideBrainSilhouette(x, y)) nodes.push({ x, y });
  }

  const maxD = 7.5;
  const edgeSet = new Set();
  const edges = [];

  const addEdge = (i, j) => {
    const a = Math.min(i, j);
    const b = Math.max(i, j);
    const key = `${a},${b}`;
    if (edgeSet.has(key)) return;
    edgeSet.add(key);
    edges.push([a, b]);
  };

  for (let i = 0; i < nodes.length; i++) {
    const dists = [];
    for (let j = i + 1; j < nodes.length; j++) {
      const dx = nodes[i].x - nodes[j].x;
      const dy = nodes[i].y - nodes[j].y;
      const d = Math.hypot(dx, dy);
      if (d < maxD) dists.push({ j, d });
    }
    dists.sort((a, b) => a.d - b.d);
    for (const { j } of dists.slice(0, 5)) addEdge(i, j);
  }

  const frontal = nodes.map((p, idx) => ({ idx, p })).filter(({ p }) => p.x < 42);
  const posterior = nodes.map((p, idx) => ({ idx, p })).filter(({ p }) => p.x > 58);
  for (let b = 0; b < 14; b++) {
    if (!frontal.length || !posterior.length) break;
    const F = frontal[(b * 7 + salt) % frontal.length].idx;
    const P = posterior[(b * 11 + salt * 3) % posterior.length].idx;
    addEdge(F, P);
  }

  return { nodes, edges };
}

function getBrainGraph() {
  if (!brainGraphLayoutCache || brainGraphLayoutVersion !== BRAIN_LAYOUT_VERSION) {
    brainGraphLayoutCache = buildBrainGraph();
    brainGraphLayoutVersion = BRAIN_LAYOUT_VERSION;
  }
  return brainGraphLayoutCache;
}

function nodeWeightsFromAttempts(byDay, nNodes) {
  const w = new Float64Array(nNodes);
  const salt = hashStr(currentUser?.email || 'anon');
  const entries = Object.entries(byDay || {});
  entries.forEach(([date, count]) => {
    const c = Math.max(0, Number(count) || 0);
    const pulses = Math.min(10, 2 + Math.ceil(c));
    for (let t = 0; t < pulses; t++) {
      const idx = (hashStr(`${date}:${t}`) + salt) % nNodes;
      w[idx] += 0.28 + Math.min(1.4, c * 0.12);
    }
  });
  let mx = 0;
  for (let i = 0; i < w.length; i++) if (w[i] > mx) mx = w[i];
  if (mx > 0) for (let i = 0; i < w.length; i++) w[i] /= mx;
  return w;
}

function synapseNodeColor(t) {
  const cold = { r: 40, g: 90, b: 180 };
  const mid = { r: 60, g: 160, b: 255 };
  const hot = { r: 160, g: 230, b: 255 };
  const a = t <= 0.5 ? t / 0.5 : (t - 0.5) / 0.5;
  const A = t <= 0.5 ? cold : mid;
  const B = t <= 0.5 ? mid : hot;
  const r = Math.round(A.r + (B.r - A.r) * a);
  const gCh = Math.round(A.g + (B.g - A.g) * a);
  const bCh = Math.round(A.b + (B.b - A.b) * a);
  return `rgb(${r},${gCh},${bCh})`;
}

function renderSynapticBrain(attemptsByDay, stats) {
  const svg = document.getElementById('synapseBrainSvg');
  const pills = document.getElementById('synapseStatPills');
  if (!svg) return;

  const { nodes, edges } = getBrainGraph();
  const w = nodeWeightsFromAttempts(attemptsByDay, nodes.length);

  const pillParts = [];
  if (stats.current_day_streak != null) {
    pillParts.push(`<span class="synapse-pill">Streak: <strong>${Number(stats.current_day_streak) || 0}</strong> d</span>`);
  }
  if (stats.best_day_streak != null) {
    pillParts.push(`<span class="synapse-pill">Best: <strong>${Number(stats.best_day_streak) || 0}</strong> d</span>`);
  }
  if (stats.distinct_practice_days != null) {
    pillParts.push(`<span class="synapse-pill">Practice days: <strong>${Number(stats.distinct_practice_days) || 0}</strong></span>`);
  }
  if (pills) pills.innerHTML = pillParts.join('');

  const edgeEls = edges.map(([i, j]) => {
    const wi = w[i];
    const wj = w[j];
    const strength = Math.sqrt((0.06 + wi) * (0.06 + wj));
    const opacity = 0.12 + strength * 0.78;
    const sw = 0.15 + strength * 0.85;
    const x1 = nodes[i].x;
    const y1 = nodes[i].y;
    const x2 = nodes[j].x;
    const y2 = nodes[j].y;
    return `<line class="synapse-edge" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke-width="${sw}" style="opacity:${opacity}" />`;
  }).join('');

  const nodeEls = nodes.map((p, idx) => {
    const t = w[idx];
    const fill = synapseNodeColor(t);
    const r = 0.55 + t * 0.95;
    const o = 0.35 + t * 0.65;
    return `<circle class="synapse-node" cx="${p.x}" cy="${p.y}" r="${r}" fill="${fill}" style="opacity:${o}" />`;
  }).join('');

  const clipId = 'synapseBrainMassClip';
  const glowId = 'synapseGlowFilter';
  const outline = brainOutlinePathD();
  const fissure = brainFissurePathD();
  const sulci = brainSulciDecorSvg();

  const outerGlowId = 'brainOuterGlow';
  svg.innerHTML = `
    <defs>
      <clipPath id="${clipId}">
        <path d="${outline}" />
      </clipPath>
      <filter id="${glowId}" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="0.45" result="b" />
        <feMerge>
          <feMergeNode in="b" />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>
      <filter id="${outerGlowId}" x="-20%" y="-20%" width="140%" height="140%">
        <feGaussianBlur stdDeviation="1.8" result="glow" />
        <feMerge>
          <feMergeNode in="glow" />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>
    </defs>
    <g filter="url(#${outerGlowId})">
      <path class="brain-silhouette-fill" d="${outline}" />
    </g>
    ${sulci}
    <path class="brain-silhouette-stroke" d="${outline}" fill="none" />
    <g clip-path="url(#${clipId})" filter="url(#${glowId})">${edgeEls}${nodeEls}</g>
    <path class="brain-fissure" d="${fissure}" fill="none" />
  `;
}

function renderAchievementBadges(badges, earnedCount, totalBadges) {
  const host = document.getElementById('achievementBadges');
  const progress = document.getElementById('badgeProgressText');
  if (progress) {
    const e = earnedCount != null ? earnedCount : (badges || []).filter(b => b.earned).length;
    const t = totalBadges != null ? totalBadges : (badges || []).length;
    progress.textContent = `${e} / ${t} unlocked`;
  }
  if (!host) return;
  if (!badges || !badges.length) {
    host.innerHTML = '<p class="badges-empty">Complete quizzes to unlock achievements.</p>';
    return;
  }
  host.innerHTML = badges.map(b => {
    const earned = !!b.earned;
    const cls = ['achievement-badge', earned ? 'achievement-badge--earned' : 'achievement-badge--locked'].join(' ');
    const icon = /^[a-z0-9_]+$/.test(String(b.icon || '')) ? b.icon : 'stars';
    return `
      <div class="${cls}" title="${esc(b.description || '')}">
        <span class="material-symbols-outlined achievement-badge-icon" aria-hidden="true">${esc(icon)}</span>
        <span class="achievement-badge-title">${esc(b.title || '')}</span>
        <span class="achievement-badge-desc">${esc(b.description || '')}</span>
      </div>
    `;
  }).join('');
}

async function loadAttemptPapers() {
  if (!currentUser) return;
  const container = document.getElementById('attemptsList');
  if (!container) return;
  if (preloadedMenuData.attempts) {
    const attempts = preloadedMenuData.attempts;
    preloadedMenuData.attempts = null;
    renderAttempts(attempts);
    return;
  }
  try {
    const res = await fetch('/api/attempts');
    const data = await res.json();
    if (res.ok && data.status === 'ok') {
      renderAttempts(data.attempts || []);
    } else {
      renderAttempts([]);
    }
  } catch (_) {
    renderAttempts([]);
  }
}

function renderAttempts(attempts) {
  const container = document.getElementById('attemptsList');
  if (!container) return;
  if (!attempts.length) {
    container.innerHTML = '<p class="attempts-empty">No attempts yet. Complete a quiz to see full paper review here.</p>';
    return;
  }

  const letters = ['A', 'B', 'C', 'D'];
  container.innerHTML = attempts.map((attempt) => {
    const created = formatSystemTime(attempt.created_at);
    const questionsHtml = (attempt.questions || []).map((q, idx) => {
      const options = q.options || [];
      const optionsHtml = options.map((opt, oi) => {
        const letter = letters[oi] || '';
        const isSelected = q.student_answer === letter;
        const isCorrect = q.correct_answer === letter;
        const classes = ['attempt-opt'];
        if (isSelected) classes.push('selected');
        if (isCorrect) classes.push('correct');
        return `<li class="${classes.join(' ')}"><strong>${letter}.</strong> ${esc(opt)}</li>`;
      }).join('');

      return `
        <div class="attempt-question ${q.is_correct ? 'correct' : 'incorrect'}">
          <div class="attempt-q-head">
            <span>Q${idx + 1}</span>
            <span>${q.is_correct ? 'Correct' : 'Incorrect'}</span>
          </div>
          <p class="attempt-q-text">${esc(q.question)}</p>
          ${questionFigureHtml(q)}
          <ul class="attempt-options">${optionsHtml}</ul>
          <p class="attempt-ans">Selected: <strong>${esc(q.student_answer || '-')}</strong>${q.student_option_text ? ` (${esc(q.student_option_text)})` : ''}</p>
          <p class="attempt-ans">Correct: <strong>${esc(q.correct_answer || '-')}</strong>${q.correct_option_text ? ` (${esc(q.correct_option_text)})` : ''}</p>
          ${q.explanation ? `<p class="attempt-exp">${esc(q.explanation)}</p>` : ''}
        </div>
      `;
    }).join('');

    return `
      <details class="attempt-card">
        <summary>
          <span>Attempt #${Number(attempt.attempt_number || attempt.attempt_id)} - ${esc(created)} (system time)</span>
          <span>${attempt.score_percent}% (${attempt.correct}/${attempt.total_questions})</span>
        </summary>
        <div class="attempt-questions">${questionsHtml}</div>
      </details>
    `;
  }).join('');
}
/* Load diagnostic quiz (single /api/quiz/load request). */
async function loadQuiz() {
  if (quizLoadInFlight) return;
  quizLoadInFlight = true;
  stopQuizTimer('diagnostic');
  hideError('quizLoadError');
  hideError('quizSubmitValidation');
  setBtn('btnLoadQuiz', true);
  document.getElementById('quizForm').classList.add('hidden');
  document.getElementById('quizResults').classList.add('hidden');
  diagnosticQuizId = '';
  diagnosticBatchIndex = 0;
  diagnosticTotalCount = 0;
  diagnosticHasMore = false;

  try {
    const loadRes = await apiFetch('/api/quiz/load', { method: 'POST' });
    const generateData = await loadRes.json();
    let timerData = generateData.timer || { enabled: false };
    if (!loadRes.ok || generateData.status !== 'ok') {
      showError(
        'quizLoadError',
        quizUserMessage(generateData, 'Diagnostic questions could not be prepared. Please try again shortly.'),
      );
      return;
    }

    quizPaper = generateData.paper || [];
    diagnosticQuizId = generateData.quiz_id || '';
    diagnosticTotalCount = Number(generateData.total_count || generateData.count || quizPaper.length || 0);
    diagnosticHasMore = generateData.has_more === true || quizPaper.length < diagnosticTotalCount;
    if (!quizPaper.length) {
      showError('quizLoadError', 'No diagnostic questions are available.');
      return;
    }
    renderQuizForm(quizPaper);
    document.getElementById('quiz-landing').classList.add('hidden');
    document.getElementById('quizForm').classList.remove('hidden');
    startQuizTimer('diagnostic', timerData);
  } catch (_) {
    showError('quizLoadError', 'Diagnostic questions could not be prepared. Please check your connection and try again.');
  }
  finally {
    quizLoadInFlight = false;
    setBtn('btnLoadQuiz', false);
  }
}

function openQuizAndLoad() {
  showPage('quiz');
}

function collectQuizAnswers(prefix, paper) {
  const answers = {};
  paper.forEach((_, i) => {
    const selected = document.querySelector(`input[name="${prefix}${i}"]:checked`);
    if (selected) answers[i] = selected.value;
  });
  return answers;
}

function restoreQuizAnswers(prefix, answers) {
  Object.entries(answers || {}).forEach(([index, value]) => {
    const input = document.querySelector(`input[name="${prefix}${index}"][value="${value}"]`);
    if (input) input.checked = true;
  });
}

function renderQuizForm(paper, preservedAnswers = {}) {
  const letters = ['A', 'B', 'C', 'D'];
  const container = document.getElementById('quizQuestions');
  container.innerHTML = paper.map((q, i) => `
    <article class="question-card ${Math.floor(i / DIAGNOSTIC_BATCH_SIZE) !== diagnosticBatchIndex ? 'hidden' : ''}" id="quiz-question-${i}" data-diagnostic-batch="${Math.floor(i / DIAGNOSTIC_BATCH_SIZE)}">
      <div class="question-num">Question ${i + 1} of ${diagnosticTotalCount || paper.length}</div>
      <p class="question-text">${esc(q.question)}</p>
      ${questionFigureHtml(q)}
      ${(q.options || []).map((opt, oi) => {
        const v = letters[oi] || '';
        return `<label class="choice-label">
          <input type="radio" name="q${i}" value="${v}" onchange="updateProgress()">
          <span class="choice-marker">${v}</span>
          <span>${esc(opt)}</span>
        </label>`;
      }).join('')}
    </article>
  `).join('');
  restoreQuizAnswers('q', preservedAnswers);
  renderDiagnosticBatch();
  updateProgress();
}

function diagnosticLoadedBatchCount() {
  return Math.max(1, Math.ceil(quizPaper.length / DIAGNOSTIC_BATCH_SIZE));
}

function diagnosticTotalBatchCount() {
  return Math.max(1, Math.ceil((diagnosticTotalCount || quizPaper.length) / DIAGNOSTIC_BATCH_SIZE));
}

function diagnosticBatchRange(batchIndex = diagnosticBatchIndex) {
  const start = Math.max(0, batchIndex * DIAGNOSTIC_BATCH_SIZE);
  const end = Math.min(quizPaper.length, start + DIAGNOSTIC_BATCH_SIZE);
  return { start, end };
}

function diagnosticBatchMissing(batchIndex = diagnosticBatchIndex) {
  const { start, end } = diagnosticBatchRange(batchIndex);
  const missing = [];
  for (let i = start; i < end; i += 1) {
    if (!document.querySelector(`input[name="q${i}"]:checked`)) missing.push(i);
  }
  return missing;
}

function renderDiagnosticBatch() {
  if (!quizPaper.length) return;
  const loadedBatches = diagnosticLoadedBatchCount();
  diagnosticBatchIndex = Math.max(0, Math.min(diagnosticBatchIndex, loadedBatches - 1));
  const { start, end } = diagnosticBatchRange();
  document.querySelectorAll('#quizQuestions .question-card[data-diagnostic-batch]').forEach((card) => {
    card.classList.toggle('hidden', Number(card.dataset.diagnosticBatch) !== diagnosticBatchIndex);
  });

  const title = document.getElementById('diagnosticBatchTitle');
  const hint = document.getElementById('diagnosticBatchHint');
  const counter = document.getElementById('diagnosticBatchCounter');
  const previous = document.getElementById('btnDiagnosticPrevious');
  const next = document.getElementById('btnDiagnosticNext');
  const submit = document.getElementById('btnSubmitQuiz');
  const missing = diagnosticBatchMissing();
  const totalBatches = diagnosticTotalBatchCount();
  const finalLoaded = !diagnosticHasMore && quizPaper.length >= diagnosticTotalCount;

  if (title) title.textContent = `Questions ${start + 1}-${end}`;
  if (counter) counter.textContent = `Set ${diagnosticBatchIndex + 1} of ${totalBatches}`;
  if (hint) {
    hint.textContent = missing.length
      ? `Answer all ${end - start} questions in this set to unlock the next set.`
      : ((diagnosticBatchIndex < loadedBatches - 1 || diagnosticHasMore)
          ? 'This set is complete. You can continue to the next five questions.'
          : 'All diagnostic questions are loaded and answered.');
  }
  if (previous) previous.classList.toggle('hidden', diagnosticBatchIndex === 0);
  if (next) {
    const canMoveOrLoad = diagnosticBatchIndex < loadedBatches - 1 || diagnosticHasMore;
    next.classList.toggle('hidden', !canMoveOrLoad);
    next.disabled = missing.length > 0;
  }
  if (submit) {
    const onFinalBatch = diagnosticBatchIndex === totalBatches - 1;
    submit.classList.toggle('hidden', !(finalLoaded && onFinalBatch));
  }
}

async function nextDiagnosticBatch() {
  const missing = diagnosticBatchMissing();
  if (missing.length) {
    focusUnansweredQuestion('diagnostic', missing[0], missing.length);
    return;
  }

  if (diagnosticBatchIndex < diagnosticLoadedBatchCount() - 1) {
    diagnosticBatchIndex += 1;
    hideError('quizSubmitValidation');
    renderDiagnosticBatch();
    document.getElementById('quizForm')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    return;
  }
  if (!diagnosticHasMore || !diagnosticQuizId) return;

  const answers = collectQuizAnswers('q', quizPaper);
  const nextButton = document.getElementById('btnDiagnosticNext');
  if (nextButton) {
    nextButton.disabled = true;
    nextButton.textContent = 'Preparing next 5...';
  }
  try {
    const res = await apiFetch('/api/quiz/next-batch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ quiz_id: diagnosticQuizId, answers }),
    });
    const data = await res.json();
    if (!res.ok || data.status !== 'ok') {
      if (data.code === 'incomplete_batch' && Number.isInteger(data.first_unanswered)) {
        focusUnansweredQuestion('diagnostic', data.first_unanswered, Number(data.unanswered_count || 1));
      } else {
        showError(
          'quizSubmitValidation',
          quizUserMessage(data, 'The next five questions are temporarily unavailable. Please try again shortly.'),
        );
      }
      return;
    }
    const batch = data.batch || [];
    if (!batch.length) {
      diagnosticHasMore = false;
      renderDiagnosticBatch();
      return;
    }
    const preserved = collectQuizAnswers('q', quizPaper);
    quizPaper = quizPaper.concat(batch);
    diagnosticTotalCount = Number(data.total_count || diagnosticTotalCount || quizPaper.length);
    diagnosticHasMore = data.has_more === true;
    diagnosticBatchIndex += 1;
    renderQuizForm(quizPaper, preserved);
    document.getElementById('quizForm')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (_) {
    showError('quizSubmitValidation', 'The next five questions are temporarily unavailable. Please try again shortly.');
  } finally {
    if (nextButton) nextButton.textContent = 'Next 5';
    renderDiagnosticBatch();
  }
}

function previousDiagnosticBatch() {
  if (diagnosticBatchIndex > 0) {
    diagnosticBatchIndex -= 1;
    hideError('quizSubmitValidation');
    renderDiagnosticBatch();
    document.getElementById('quizForm')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

function updateProgress() {
  const answered = quizPaper.filter((_, i) => document.querySelector(`input[name="q${i}"]:checked`)).length;
  const denominator = diagnosticTotalCount || quizPaper.length;
  const pct = denominator ? (answered / denominator) * 100 : 0;
  document.getElementById('quizProgressBar').style.width = `${pct}%`;
  quizPaper.forEach((_, i) => {
    if (document.querySelector(`input[name="q${i}"]:checked`)) {
      document.getElementById(`quiz-question-${i}`)?.classList.remove('question-needs-answer');
    }
  });
  renderDiagnosticBatch();
  if (!diagnosticHasMore && answered === denominator) hideError('quizSubmitValidation');
}

async function submitQuiz(event, autoSubmit = false) {
  if (event) event.preventDefault();
  if (!diagnosticQuizId) return;

  if (!autoSubmit && diagnosticHasMore) {
    showError('quizSubmitValidation', 'Complete the remaining question sets before submitting.');
    return;
  }

  if (!autoSubmit && diagnosticTimerState.enabled) {
    const missing = unansweredQuestionIndices(quizPaper, 'q');
    if (missing.length) {
      focusUnansweredQuestion('diagnostic', missing[0], missing.length);
      return;
    }
  }

  hideError('quizSubmitValidation');
  setBtn('btnSubmitQuiz', true);
  const answers = {};
  quizPaper.forEach((_, i) => {
    const sel = document.querySelector(`input[name="q${i}"]:checked`);
    if (sel) answers[i] = sel.value;
  });
  try {
    const res  = await apiFetch('/api/submit', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ quiz_id: diagnosticQuizId, answers }) });
    const data = await res.json();
    if (!res.ok || data.status !== 'ok') {
      if (data.code === 'incomplete_quiz' && Number.isInteger(data.first_unanswered)) {
        focusUnansweredQuestion('diagnostic', data.first_unanswered, Number(data.unanswered_count || 1));
      } else {
        showError('quizSubmitValidation', data.message || 'Submission failed.');
      }
      return;
    }
    stopQuizTimer('diagnostic');
    renderResults(data);
    diagnosticQuizId = '';
    document.getElementById('quizForm').classList.add('hidden');
    document.getElementById('quizResults').classList.remove('hidden');
    loadDashboard();
    if (currentPage === 'attempts') loadAttemptPapers();
  } catch (e) {
    showError('quizSubmitValidation', e.message);
  } finally {
    setBtn('btnSubmitQuiz', false);
  }
}

function renderResults(data) {
  const pct = data.score_percent ?? 0;
  document.getElementById('scorePct').textContent    = `${pct}%`;
  document.getElementById('scoreSubText').textContent = `${data.correct}/${data.total}`;

  // Animate ring
  const circ = 2 * Math.PI * 50;
  const fill = (pct / 100) * circ;
  document.getElementById('scoreRingFill').setAttribute('stroke-dasharray', `${fill} ${circ}`);

  const grade = pct >= 80 ? 'Excellent!' : pct >= 60 ? 'Good work!' : 'Keep going!';
  document.getElementById('resultHeading').textContent    = grade;
  document.getElementById('resultSubHeading').textContent = `Score: ${pct}% — ${data.correct} correct, ${data.incorrect} incorrect`;

  const reviewList = document.getElementById('reviewList');
  reviewList.innerHTML = (data.review || []).map((item, i) => `
    <div class="review-card ${item.is_correct ? 'correct' : 'incorrect'}">
      <div class="review-status">${item.is_correct ? 'Correct' : 'Incorrect'}</div>
      <p class="review-question">Q${i + 1}. ${esc(item.question)}</p>
      ${questionFigureHtml(item)}
      <p class="review-answer">Your answer: <strong>${esc(item.student_answer || '—')}</strong></p>
      ${!item.is_correct ? `<p class="review-answer">Correct answer: <strong>${esc(item.correct_answer)}</strong></p>` : ''}
      ${item.explanation ? `<p class="review-explanation">${esc(item.explanation)}</p>` : ''}
    </div>
  `).join('');
}

function resetQuiz() {
  stopQuizTimer('diagnostic');
  quizPaper = [];
  diagnosticQuizId = '';
  diagnosticBatchIndex = 0;
  diagnosticTotalCount = 0;
  diagnosticHasMore = false;
  document.getElementById('quizResults').classList.add('hidden');
  document.getElementById('quizForm').classList.add('hidden');
  document.getElementById('quizLoadError').classList.add('hidden');
  document.getElementById('quiz-landing').classList.remove('hidden');
  document.getElementById('quizProgressBar').style.width = '0%';
  hideError('quizSubmitValidation');
}

/* Guidance page */
async function loadGuidance() {
  hideError('guidanceError');
  setBtn('btnGuidance', true);
  document.getElementById('guidanceContent').classList.add('hidden');
  try {
    if (preloadedMenuData.guidance) {
      const data = preloadedMenuData.guidance;
      preloadedMenuData.guidance = null;
      populateGuidanceAttempts(data.attempts || [], data.selected_attempt_id);
      renderGuidance(data);
      document.getElementById('guidanceContent').classList.remove('hidden');
      return;
    }
    const attemptSelect = document.getElementById('guidanceAttemptSelect');
    const selectedAttemptId = attemptSelect && attemptSelect.value ? Number(attemptSelect.value) : null;
    const query = selectedAttemptId ? `?attempt_id=${encodeURIComponent(selectedAttemptId)}` : '';
    const res  = await fetch(`/api/recommend${query}`);
    const data = await res.json();
    if (!res.ok || data.status !== 'ok') { showError('guidanceError', data.message || 'Failed to load guidance.'); return; }
    populateGuidanceAttempts(data.attempts || [], data.selected_attempt_id);
    renderGuidance(data);
    document.getElementById('guidanceContent').classList.remove('hidden');
  } catch (e) { showError('guidanceError', e.message); }
  finally { setBtn('btnGuidance', false); }
}

function populateGuidanceAttempts(attempts, selectedAttemptId) {
  const select = document.getElementById('guidanceAttemptSelect');
  if (!select) return;
  if (!attempts.length) {
    select.innerHTML = '<option value="">No attempts yet</option>';
    select.disabled = true;
    return;
  }

  select.disabled = false;
  select.innerHTML = attempts.map(a => {
    const selected = Number(a.attempt_id) === Number(selectedAttemptId) ? ' selected' : '';
    return `<option value="${Number(a.attempt_id)}"${selected}>${esc(a.label || `Attempt ${a.attempt_id}`)}</option>`;
  }).join('');
}

function renderGuidance(data) {
  // Score trend sparkline
  const trend  = data.score_trend || [];
  const canvas = document.getElementById('trendCanvas');
  const ctx    = canvas.getContext('2d');
  canvas.width = canvas.parentElement.clientWidth - 48 || 260;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (trend.length >= 2) {
    const max = Math.max(...trend, 100);
    const min = Math.min(...trend, 0);
    const px  = i => (i / (trend.length - 1)) * canvas.width;
    const py  = v => canvas.height - ((v - min) / (max - min + 1)) * canvas.height;

    const grad = ctx.createLinearGradient(0, 0, canvas.width, 0);
    grad.addColorStop(0, '#6c63ff');
    grad.addColorStop(1, '#8b5cf6');
    ctx.strokeStyle  = grad;
    ctx.lineWidth    = 2.5;
    ctx.lineJoin     = 'round';
    ctx.beginPath();
    trend.forEach((v, i) => i === 0 ? ctx.moveTo(px(i), py(v)) : ctx.lineTo(px(i), py(v)));
    ctx.stroke();

    // Dots
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--focus-green').trim() || '#3ecf8e';
    trend.forEach((v, i) => { ctx.beginPath(); ctx.arc(px(i), py(v), 4, 0, Math.PI * 2); ctx.fill(); });
  } else {
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--ink3').trim() || '#9aa3b8';
    ctx.font = '0.85rem Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('No attempts yet', canvas.width / 2, canvas.height / 2);
  }

  const label = trend.length ? `Scores: ${trend.join(' -> ')}` : 'No attempts yet';
  document.getElementById('trendLabel').textContent = label;

  // Focus areas: ranked weak topics + Bloom-level mix (not chapter paths)
  const weakList = document.getElementById('weakTopicsList');
  weakList.innerHTML = (data.weak_topics || []).length
    ? data.weak_topics.map((t, idx) => {
      const br = t.difficulty_breakdown
        ? `<div class="topic-meta">Bloom levels: ${esc(t.difficulty_breakdown)}</div>`
        : '';
      return `<li class="topic-item"><span class="topic-rank" aria-hidden="true">${idx + 1}</span><span class="topic-dot"></span><div class="topic-item-text"><div class="topic-line"><strong>${esc(t.topic_name || t.topic_id)}</strong> — ${Number(t.mistakes || 0)} mistake(s)</div>${br}</div></li>`;
    }).join('')
    : '<li style="color:var(--ink3);font-size:.9rem">No weak topics identified yet.</li>';

  // Recommendations: summary paragraphs + per-topic practice actions (grades/chapters)
  const summaryEl = document.getElementById('guidanceSummary');
  const notesList = document.getElementById('guidanceNotesList');
  const summaries = (data.guidance_notes || []).map(n => `<p class="guidance-summary-p">${esc(n)}</p>`).join('');
  if (summaryEl) summaryEl.innerHTML = summaries;

  const recs = data.study_recommendations || [];
  const actionItems = recs.map((r) => {
    const actions = (r.practice_actions || []).length
      ? `<ul class="reco-actions">${r.practice_actions.map(a => `<li>${esc(a)}</li>`).join('')}</ul>`
      : '<p class="reco-fallback">No mapped textbook chapters for this topic.</p>';
    const tip = r.tip ? `<div class="reco-tip">${esc(r.tip)}</div>` : '';
    return `
      <li class="note-item reco-item">
        <span class="material-symbols-outlined note-item-icon" aria-hidden="true">school</span>
        <div class="reco-body">
          <div class="reco-head">
            <strong>${esc(r.topic_name || r.topic_id)}</strong>
            <span class="reco-pill">${Number(r.mistakes || 0)} error(s)</span>
          </div>
          ${tip}
          ${actions}
        </div>
      </li>
    `;
  });

  if (recs.length) {
    notesList.innerHTML = actionItems.join('');
  } else {
    notesList.innerHTML = '<li style="color:var(--ink3);font-size:.9rem">Complete a quiz with incorrect answers to see practice targets here.</li>';
  }

}

async function startPracticeQuiz(grade, topic) {
  stopQuizTimer('practice');
  hideError('practiceError');
  hideError('practiceQuizLoadError');
  hideError('practiceSubmitValidation');
  setPracticeLoading(true, `Loading questions for Grade ${grade} - ${topic}...`);
  quizLoadInFlight = true;
  document.getElementById('practiceQuizHint').classList.add('hidden');
  document.getElementById('practiceQuizForm').classList.add('hidden');
  document.getElementById('practiceQuizResults').classList.add('hidden');
  practiceQuizId = '';
  practiceBatchIndex = 0;
  practiceTotalCount = 0;
  practiceHasMore = false;
  try {
    const res = await apiFetch('/api/practice/quiz', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ grade, topic }),
    });
    const data = await res.json();
    if (!res.ok || data.status !== 'ok') {
      document.getElementById('practiceCatalogView')?.classList.remove('hidden');
      document.getElementById('practiceQuizPage')?.classList.add('hidden');
      showError(
        'practiceQuizLoadError',
        quizUserMessage(data, 'Practice questions could not be prepared. Please try again shortly.'),
      );
      return;
    }

    practicePaper = data.paper || [];
    practiceQuizId = data.quiz_id || '';
    practiceTotalCount = Number(data.total_count || 10);
    practiceHasMore = data.has_more === true || practicePaper.length < practiceTotalCount;
    if (!practicePaper.length) {
      showError('practiceQuizLoadError', 'No practice questions are available for this topic.');
      return;
    }

    const title = document.getElementById('practiceQuizPageTitle');
    const subtitle = document.getElementById('practiceQuizPageSubtitle');
    if (title) title.textContent = topic;
    if (subtitle) subtitle.textContent = `Grade ${grade} · ${practiceTotalCount} questions`;
    document.getElementById('practiceCatalogView')?.classList.add('hidden');
    document.getElementById('practiceQuizPage')?.classList.remove('hidden');
    renderPracticeQuizForm(practicePaper);
    document.getElementById('practiceQuizForm').classList.remove('hidden');
    startQuizTimer('practice', data.timer || { enabled: false });
    document.getElementById('practiceQuizPage')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (_) {
    document.getElementById('practiceCatalogView')?.classList.remove('hidden');
    document.getElementById('practiceQuizPage')?.classList.add('hidden');
    showError('practiceQuizLoadError', 'Practice questions could not be prepared. Please check your connection and try again.');
  } finally {
    setPracticeLoading(false);
    quizLoadInFlight = false;
  }
}

async function loadPracticeCatalog() {
  hideError('practiceError');
  try {
    let data = preloadedMenuData.practice;
    if (data) {
      preloadedMenuData.practice = null;
    } else {
      const res = await fetch('/api/practice/catalog');
      data = await res.json();
      if (!res.ok || data.status !== 'ok') {
        showError('practiceError', data.message || 'Failed to load practice topics.');
        return;
      }
    }
    practiceTabs = data.tabs || [];
    if (!practiceTabs.length) {
      document.getElementById('practiceTopicTabs').innerHTML = '';
      document.getElementById('practiceTopicPanel').innerHTML = '<p style="color:var(--ink3);font-size:.9rem">No practice topics available.</p>';
      return;
    }
    if (!practiceTabs.some(t => t.tab_id === activePracticeTabId)) {
      activePracticeTabId = practiceTabs[0].tab_id;
    }
    renderPracticeTabs();
    renderPracticePanel(activePracticeTabId);
  } catch (e) {
    showError('practiceError', e.message);
  }
}

function renderPracticeTabs() {
  const tabsEl = document.getElementById('practiceTopicTabs');
  if (!tabsEl) return;
  tabsEl.innerHTML = (practiceTabs || []).map(tab => `
    <button type="button" class="practice-tab-btn ${tab.tab_id === activePracticeTabId ? 'active' : ''}" data-tab-id="${esc(tab.tab_id)}">
      ${esc(tab.label)}
    </button>
  `).join('');

  tabsEl.querySelectorAll('.practice-tab-btn[data-tab-id]').forEach((btn) => {
    btn.onclick = () => {
      activePracticeTabId = btn.getAttribute('data-tab-id') || '';
      renderPracticeTabs();
      renderPracticePanel(activePracticeTabId);
    };
  });
}

function renderPracticePanel(tabId) {
  const panelEl = document.getElementById('practiceTopicPanel');
  if (!panelEl) return;
  const tab = (practiceTabs || []).find(t => t.tab_id === tabId);
  if (!tab) {
    panelEl.innerHTML = '<p style="color:var(--ink3);font-size:.9rem">Select a topic tab.</p>';
    return;
  }

  panelEl.innerHTML = (tab.grades || []).map(g => {
    const buttons = (g.subtopics || []).map(s => {
      const topic = String(s.topic || '');
      const topicEncoded = encodeURIComponent(String(s.topic || ''));
      const isActive = Number(g.grade) === Number(selectedPracticeGrade) && topic === selectedPracticeTopic;
      return `
        <button type="button" class="practice-subtopic-btn ${isActive ? 'active' : ''}" data-grade="${Number(g.grade)}" data-topic="${topicEncoded}">
          ${esc(s.topic)}
        </button>
      `;
    }).join('');
    return `
      <div class="practice-grade-block">
        <div class="practice-grade-title">Grade ${Number(g.grade)}</div>
        <div class="practice-subtopic-grid">${buttons}</div>
      </div>
    `;
  }).join('');

  panelEl.querySelectorAll('.practice-subtopic-btn[data-grade][data-topic]').forEach((btn) => {
    btn.onclick = () => {
      const grade = Number(btn.getAttribute('data-grade') || '0');
      const topic = decodeURIComponent(btn.getAttribute('data-topic') || '');
      selectedPracticeGrade = grade;
      selectedPracticeTopic = topic;
      renderPracticePanel(activePracticeTabId);
      document.getElementById('practiceQuizHint').scrollIntoView({ behavior: 'smooth', block: 'start' });
      startPracticeQuiz(grade, topic);
    };
  });
}

function renderPracticeQuizForm(paper, preservedAnswers = {}) {
  const letters = ['A', 'B', 'C', 'D'];
  const container = document.getElementById('practiceQuizQuestions');
  container.innerHTML = paper.map((q, i) => `
    <article class="question-card ${Math.floor(i / PRACTICE_BATCH_SIZE) !== practiceBatchIndex ? 'hidden' : ''}" id="practice-question-${i}" data-practice-batch="${Math.floor(i / PRACTICE_BATCH_SIZE)}">
      <div class="question-num">Question ${i + 1} of ${practiceTotalCount || paper.length}</div>
      <p class="question-text">${esc(q.question)}</p>
      ${questionFigureHtml(q)}
      ${(q.options || []).map((opt, oi) => {
        const v = letters[oi] || '';
        return `<label class="choice-label">
          <input type="radio" name="pq${i}" value="${v}" onchange="updatePracticeProgress()">
          <span class="choice-marker">${v}</span>
          <span>${esc(opt)}</span>
        </label>`;
      }).join('')}
    </article>
  `).join('');
  restoreQuizAnswers('pq', preservedAnswers);
  renderPracticeBatch();
  updatePracticeProgress();
}

function practiceLoadedBatchCount() {
  return Math.max(1, Math.ceil(practicePaper.length / PRACTICE_BATCH_SIZE));
}

function practiceTotalBatchCount() {
  return Math.max(1, Math.ceil((practiceTotalCount || practicePaper.length) / PRACTICE_BATCH_SIZE));
}

function practiceBatchRange(batchIndex = practiceBatchIndex) {
  const start = Math.max(0, batchIndex * PRACTICE_BATCH_SIZE);
  const end = Math.min(practicePaper.length, start + PRACTICE_BATCH_SIZE);
  return { start, end };
}

function practiceBatchMissing(batchIndex = practiceBatchIndex) {
  const { start, end } = practiceBatchRange(batchIndex);
  const missing = [];
  for (let i = start; i < end; i += 1) {
    if (!document.querySelector(`input[name="pq${i}"]:checked`)) missing.push(i);
  }
  return missing;
}

function renderPracticeBatch() {
  if (!practicePaper.length) return;
  const loadedBatches = practiceLoadedBatchCount();
  practiceBatchIndex = Math.max(0, Math.min(practiceBatchIndex, loadedBatches - 1));
  const { start, end } = practiceBatchRange();

  document.querySelectorAll('#practiceQuizQuestions .question-card[data-practice-batch]').forEach((card) => {
    card.classList.toggle('hidden', Number(card.dataset.practiceBatch) !== practiceBatchIndex);
  });

  const title = document.getElementById('practiceBatchTitle');
  const hint = document.getElementById('practiceBatchHint');
  const counter = document.getElementById('practiceBatchCounter');
  const previous = document.getElementById('btnPracticePrevious');
  const next = document.getElementById('btnPracticeNext');
  const submit = document.getElementById('btnSubmitPractice');
  const missing = practiceBatchMissing();
  const totalBatches = practiceTotalBatchCount();
  const finalLoaded = !practiceHasMore && practicePaper.length >= practiceTotalCount;

  if (title) title.textContent = `Questions ${start + 1}-${end}`;
  if (counter) counter.textContent = `Set ${practiceBatchIndex + 1} of ${totalBatches}`;
  if (hint) {
    hint.textContent = missing.length
      ? `Answer all ${end - start} questions in this set to unlock the next set.`
      : ((practiceBatchIndex < loadedBatches - 1 || practiceHasMore)
          ? 'This set is complete. You can continue to the next five questions.'
          : 'All questions in this final set are answered.');
  }
  if (previous) previous.classList.toggle('hidden', practiceBatchIndex === 0);
  if (next) {
    const canMoveOrLoad = practiceBatchIndex < loadedBatches - 1 || practiceHasMore;
    next.classList.toggle('hidden', !canMoveOrLoad);
    next.disabled = missing.length > 0;
  }
  if (submit) {
    const onFinalBatch = practiceBatchIndex === totalBatches - 1;
    submit.classList.toggle('hidden', !(finalLoaded && onFinalBatch));
  }
}

async function nextPracticeBatch() {
  const missing = practiceBatchMissing();
  if (missing.length) {
    focusUnansweredQuestion('practice', missing[0], missing.length);
    return;
  }

  if (practiceBatchIndex < practiceLoadedBatchCount() - 1) {
    practiceBatchIndex += 1;
    hideError('practiceSubmitValidation');
    renderPracticeBatch();
    document.getElementById('practiceQuizForm')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    return;
  }
  if (!practiceHasMore || !practiceQuizId) return;

  const answers = collectQuizAnswers('pq', practicePaper);
  setPracticeLoading(true, 'Preparing the next 5 questions...');
  try {
    const res = await apiFetch('/api/quiz/next-batch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ quiz_id: practiceQuizId, answers }),
    });
    const data = await res.json();
    if (!res.ok || data.status !== 'ok') {
      if (data.code === 'incomplete_batch' && Number.isInteger(data.first_unanswered)) {
        focusUnansweredQuestion('practice', data.first_unanswered, Number(data.unanswered_count || 1));
      } else {
        showError(
          'practiceSubmitValidation',
          quizUserMessage(data, 'The next five questions are temporarily unavailable. Please try again shortly.'),
        );
      }
      return;
    }
    const batch = data.batch || [];
    if (!batch.length) {
      practiceHasMore = false;
      renderPracticeBatch();
      return;
    }
    const preserved = collectQuizAnswers('pq', practicePaper);
    practicePaper = practicePaper.concat(batch);
    practiceTotalCount = Number(data.total_count || practiceTotalCount || practicePaper.length);
    practiceHasMore = data.has_more === true;
    practiceBatchIndex += 1;
    renderPracticeQuizForm(practicePaper, preserved);
    document.getElementById('practiceQuizForm')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (_) {
    showError('practiceSubmitValidation', 'The next five questions are temporarily unavailable. Please try again shortly.');
  } finally {
    setPracticeLoading(false);
    renderPracticeBatch();
  }
}

function previousPracticeBatch() {
  if (practiceBatchIndex > 0) {
    practiceBatchIndex -= 1;
    hideError('practiceSubmitValidation');
    renderPracticeBatch();
    document.getElementById('practiceQuizForm')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

function backToPracticeTopics() {
  stopQuizTimer('practice');
  document.getElementById('practiceQuizPage')?.classList.add('hidden');
  document.getElementById('practiceCatalogView')?.classList.remove('hidden');
  document.getElementById('practiceQuizForm')?.classList.add('hidden');
  document.getElementById('practiceQuizResults')?.classList.add('hidden');
  document.getElementById('practiceQuizHint')?.classList.remove('hidden');
  practiceBatchIndex = 0;
  practiceTotalCount = 0;
  practiceHasMore = false;
  hideError('practiceSubmitValidation');
}

function updatePracticeProgress() {
  const answered = practicePaper.filter((_, i) => document.querySelector(`input[name="pq${i}"]:checked`)).length;
  const denominator = practiceTotalCount || practicePaper.length;
  const pct = denominator ? (answered / denominator) * 100 : 0;
  document.getElementById('practiceQuizProgressBar').style.width = `${pct}%`;
  practicePaper.forEach((_, i) => {
    if (document.querySelector(`input[name="pq${i}"]:checked`)) {
      document.getElementById(`practice-question-${i}`)?.classList.remove('question-needs-answer');
    }
  });
  renderPracticeBatch();
  if (!practiceHasMore && answered === denominator) hideError('practiceSubmitValidation');
}

async function submitPracticeQuiz(event, autoSubmit = false) {
  if (event) event.preventDefault();
  if (!practiceQuizId) return;

  if (!autoSubmit && practiceHasMore) {
    showError('practiceSubmitValidation', 'Complete the remaining question sets before submitting.');
    return;
  }

  if (!autoSubmit && practiceTimerState.enabled) {
    const missing = unansweredQuestionIndices(practicePaper, 'pq');
    if (missing.length) {
      focusUnansweredQuestion('practice', missing[0], missing.length);
      return;
    }
  }

  hideError('practiceSubmitValidation');
  setBtn('btnSubmitPractice', true);
  const answers = {};
  practicePaper.forEach((_, i) => {
    const sel = document.querySelector(`input[name="pq${i}"]:checked`);
    if (sel) answers[i] = sel.value;
  });
  try {
    const res = await apiFetch('/api/submit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ quiz_id: practiceQuizId, answers }),
    });
    const data = await res.json();
    if (!res.ok || data.status !== 'ok') {
      if (data.code === 'incomplete_quiz' && Number.isInteger(data.first_unanswered)) {
        focusUnansweredQuestion('practice', data.first_unanswered, Number(data.unanswered_count || 1));
      } else {
        showError('practiceSubmitValidation', data.message || 'Practice submission failed.');
      }
      return;
    }
    stopQuizTimer('practice');
    renderPracticeResults(data);
    practiceQuizId = '';
    document.getElementById('practiceQuizForm').classList.add('hidden');
    document.getElementById('practiceQuizResults').classList.remove('hidden');
    loadDashboard();
    if (currentPage === 'attempts') loadAttemptPapers();
  } catch (e) {
    showError('practiceSubmitValidation', e.message);
  } finally {
    setBtn('btnSubmitPractice', false);
  }
}

function renderPracticeResults(data) {
  const pct = Number(data.score_percent ?? 0);
  document.getElementById('practiceResultHeading').textContent = pct >= 80 ? 'Excellent Practice' : pct >= 60 ? 'Good Practice' : 'Keep Practicing';
  document.getElementById('practiceResultSubHeading').textContent = `${data.correct} correct, ${data.incorrect} incorrect`;
  document.getElementById('practiceScoreText').textContent = `Score: ${pct}% (${data.correct}/${data.total})`;

  const reviewList = document.getElementById('practiceReviewList');
  reviewList.innerHTML = (data.review || []).map((item, i) => `
    <div class="review-card ${item.is_correct ? 'correct' : 'incorrect'}">
      <div class="review-status">${item.is_correct ? 'Correct' : 'Incorrect'}</div>
      <p class="review-question">Q${i + 1}. ${esc(item.question)}</p>
      ${questionFigureHtml(item)}
      <p class="review-answer">Your answer: <strong>${esc(item.student_answer || '-')}</strong></p>
      ${!item.is_correct ? `<p class="review-answer">Correct answer: <strong>${esc(item.correct_answer)}</strong></p>` : ''}
      ${item.explanation ? `<p class="review-explanation">${esc(item.explanation)}</p>` : ''}
    </div>
  `).join('');
}

function resetPracticeQuiz() {
  stopQuizTimer('practice');
  practicePaper = [];
  practiceQuizId = '';
  practiceBatchIndex = 0;
  practiceTotalCount = 0;
  practiceHasMore = false;
  document.getElementById('practiceQuizResults').classList.add('hidden');
  document.getElementById('practiceQuizForm').classList.add('hidden');
  document.getElementById('practiceQuizPage')?.classList.add('hidden');
  document.getElementById('practiceCatalogView')?.classList.remove('hidden');
  document.getElementById('practiceQuizHint').classList.remove('hidden');
  document.getElementById('practiceQuizProgressBar').style.width = '0%';
  hideError('practiceQuizLoadError');
  hideError('practiceSubmitValidation');
  loadPracticeCatalog();
}

/* Settings */
async function loadSettings() {
  try {
    let data = preloadedMenuData.settings;
    if (data) {
      preloadedMenuData.settings = null;
    } else {
      const res = await fetch('/api/settings');
      data = await res.json();
    }
    if (data.status === 'ok') {
      applyTheme(data.theme || 'dark');
      document.getElementById('themeToggle').checked = (data.theme === 'dark');
      const diff = document.getElementById('difficultySelect');
      if (diff) diff.value = data.difficulty || 'medium';
      const timerToggle = document.getElementById('quizTimerToggle');
      if (timerToggle) timerToggle.checked = data.quiz_timer_enabled !== false;
    }
  } catch (_) {}

  try {
    let data = preloadedMenuData.profile;
    if (data) {
      preloadedMenuData.profile = null;
    } else {
      const res = await fetch('/api/profile');
      data = await res.json();
    }
    if (data.status === 'ok' && data.name) {
      document.getElementById('settingsName').value = data.name;
    }
  } catch (_) {}
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
}

function toggleTheme(checkbox) {
  applyTheme(checkbox.checked ? 'dark' : 'light');
  saveSettings();
}

async function saveName() {
  const name = document.getElementById('settingsName').value.trim();
  if (!name) return;
  try {
    const res  = await apiFetch('/api/profile', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ name }) });
    const data = await res.json();
    if (data.status === 'ok') {
      flashOk('nameStatus');
      if (currentUser) {
        currentUser.name = name;
        applyUserToUI();
        document.getElementById('welcomeHeading').textContent = `Welcome back, ${name}!`;
      }
    }
  } catch (_) {}
}

async function saveSettings() {
  const theme      = document.getElementById('themeToggle').checked ? 'dark' : 'light';
  const difficulty = document.getElementById('difficultySelect').value;
  const quizTimerEnabled = document.getElementById('quizTimerToggle')?.checked !== false;
  try {
    await apiFetch('/api/settings', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ theme, difficulty, quiz_timer_enabled: quizTimerEnabled }),
    });
    flashOk('settingsStatus');
  } catch (_) {}
}

/* Init */
async function init() {
  await handleAuthLink();
  try {
    const res  = await fetch('/api/me');
    const data = await res.json();
    if (data.email) {
      resetMenuPreloadCache();
      currentUser = { email: data.email, name: data.name || data.email };
      applyUserToUI();
      document.getElementById('auth-overlay').style.display = 'none';
      document.getElementById('app').classList.remove('hidden');
      applyStoredSidebarLayout();
      showPage('dashboard');
      void preloadAuthenticatedMenuData();
    }
  } catch (_) {}
}

// Settings tab: load when navigating to it
document.getElementById('nav-settings').addEventListener('click', loadSettings);

window.addEventListener('resize', () => {
  const sidebar = document.getElementById('sidebar');
  const backdrop = document.getElementById('sidebarBackdrop');
  const app = document.getElementById('app');
  const drawerLayout = window.matchMedia('(max-width: 1024px)').matches;
  if (!drawerLayout) {
    if (sidebar) sidebar.classList.remove('open');
    if (backdrop) backdrop.classList.remove('open');
  }
  if (!window.matchMedia('(min-width: 1025px)').matches) {
    app?.classList.remove('sidebar-collapsed');
  } else {
    try {
      if (localStorage.getItem('synapSidebarCollapsed') === '1') {
        app?.classList.add('sidebar-collapsed');
      }
    } catch (_) {}
  }
  syncSidebarPinUi();
  syncMobileMenuButton();
});

document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  const sidebar = document.getElementById('sidebar');
  if (!sidebar || !sidebar.classList.contains('open')) return;
  sidebar.classList.remove('open');
  document.getElementById('sidebarBackdrop')?.classList.remove('open');
  syncMobileMenuButton();
  syncSidebarPinUi();
  document.getElementById('mobileMenuBtn')?.focus();
});

enableAuthEnterSubmission();
init();
