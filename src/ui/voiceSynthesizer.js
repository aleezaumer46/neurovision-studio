/**
 * Module 9: Voice Accessibility & Audio Synthesis
 * Powered by Web Speech API (window.speechSynthesis)
 */

export class VoiceSynthesizer {
  constructor() {
    this.isSupported = 'speechSynthesis' in window;
    this.isEnabled = localStorage.getItem('neurovision_voice_enabled') === 'true';
    this.rate = 1.0;
    this.pitch = 1.0;
    this.selectedVoice = null;
    this.lastSpokenTime = 0;
    this.debounceMs = 1800; // minimum 1.8s between announcements to prevent audio queue overlap

    if (this.isSupported) {
      window.speechSynthesis.onvoiceschanged = () => {
        this.loadVoices();
      };
      this.loadVoices();
    }
  }

  loadVoices() {
    if (!this.isSupported) return;
    const voices = window.speechSynthesis.getVoices();
    // Prefer natural English voices (Google US English, Samantha, Microsoft Zira, etc.)
    this.selectedVoice = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Online')))
      || voices.find(v => v.lang.startsWith('en'))
      || voices[0];
  }

  toggle() {
    this.isEnabled = !this.isEnabled;
    localStorage.setItem('neurovision_voice_enabled', this.isEnabled ? 'true' : 'false');
    if (this.isEnabled) {
      this.speak('Voice assistant activated', true);
    } else {
      this.cancel();
    }
    return this.isEnabled;
  }

  speak(text, force = false) {
    if (!this.isSupported || !this.isEnabled || !text) return;

    const now = Date.now();
    if (!force && (now - this.lastSpokenTime < this.debounceMs)) {
      return; // prevent overlapping speech queue
    }
    this.lastSpokenTime = now;

    try {
      this.cancel(); // cancel pending utterances
      const utterance = new SpeechSynthesisUtterance(text);
      if (this.selectedVoice) utterance.voice = this.selectedVoice;
      utterance.rate = this.rate;
      utterance.pitch = this.pitch;
      window.speechSynthesis.speak(utterance);
    } catch (err) {
      console.warn('Speech synthesis error:', err);
    }
  }

  announceVerdict(className, confidencePct, isOOD = false) {
    if (!this.isEnabled) return;
    const cleanName = className.replace(/[^a-zA-Z0-9 /]/g, '').trim();

    if (isOOD) {
      this.speak(`Input is uncertain or out of distribution. Nearest category is ${cleanName} at ${confidencePct} percent.`);
    } else {
      this.speak(`Detected ${cleanName} with ${confidencePct} percent confidence.`);
    }
  }

  cancel() {
    if (this.isSupported) {
      window.speechSynthesis.cancel();
    }
  }
}

export const voiceSynthesizer = new VoiceSynthesizer();
