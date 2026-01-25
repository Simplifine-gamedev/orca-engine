import * as React from 'react';

interface HotkeyConfig {
  key: string;
  callback: () => void;
  description?: string;
}

export const useHotkeys = (hotkeys: HotkeyConfig[]) => {
  React.useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      // Don't trigger hotkeys if user is typing in an input
      if (
        event.target instanceof HTMLInputElement ||
        event.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      const matchingHotkey = hotkeys.find(
        (hotkey) => hotkey.key.toLowerCase() === event.key.toLowerCase()
      );

      if (matchingHotkey) {
        event.preventDefault();
        matchingHotkey.callback();
      }
    };

    window.addEventListener('keydown', handleKeyPress);

    return () => {
      window.removeEventListener('keydown', handleKeyPress);
    };
  }, [hotkeys]);
};
