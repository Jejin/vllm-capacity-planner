import App from './App.svelte';
import { mount } from 'svelte';
import { inject } from '@vercel/analytics';

// Inject Vercel Analytics
inject({ mode: import.meta.env.DEV ? 'development' : 'production' });

const app = mount(App, { target: document.getElementById('app')! });
export default app;
