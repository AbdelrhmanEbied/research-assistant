// Shared DOM references + application state. Modules import the bits they
// need from here rather than each grabbing their own copy of the same node.

export const dom = {
  messages: document.getElementById('messages'),
  convList: document.getElementById('convList'),
  input: document.getElementById('input'),
  sendBtn: document.getElementById('sendBtn'),
  newChatBtn: document.getElementById('newChatBtn'),
  mainHeader: document.getElementById('mainHeader'),
  headerActions: document.getElementById('headerActions'),
  toast: document.getElementById('toast'),
  sidebar: document.getElementById('sidebar'),
  sidebarToggle: document.getElementById('sidebarToggle'),
  sourceSelect: document.getElementById('sourceSelect'),
};

// captured before anything mutates it (splitHeroLetters rewrites the
// headline), so the empty state only exists in one place: the markup
export const EMPTY_STATE_HTML = dom.messages.innerHTML;

export const state = {
  currentConversationId: null,
  isStreaming: false,
  conversationsCache: [],
  messagesPage: { total: 0, loaded: 0, limit: 200 },
  currentController: null,
  // true only when the user pressed the stop button; a stream that ends or
  // breaks for any other reason is not a user stop
  userStopped: false,
  pendingMode: null,
  agentMode: 'fast',
};