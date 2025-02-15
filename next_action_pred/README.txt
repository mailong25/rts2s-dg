Format for each sample (conversation):
 - id: The ID of the conversation (e.g., sw_20000).
 - action: The next action for every 160ms. '0' for speaker 0 (e.g., sw_20000_0.wav), '1' for speaker 1. The audio len for each speaker is 5 min, so there will be 1875 next action labels for each.
    - Remain silent (3)
    - Continue speaking (4)
    - Start speaking (5)
    - Stop speaking (6)
 - vad: Timestamps for voice segments detected by a Voice Activity Detector (VAD).
----------------
Note:
You might see consecutive start_speaking labels (e.g., [5, 5, 5]) in the action labels.
This occurs because the action start_speaking can be true across multiple sliding windows.
For example, if a speaker starts speaking at 3.1 seconds, it is possible to predict start_speaking at timestamps like 3.00, 2.84, or 2.68.
The same logic applies to the stop_speaking label.