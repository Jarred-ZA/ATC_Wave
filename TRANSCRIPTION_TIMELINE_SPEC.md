# Transcription Timeline Feature Specification

## Overview
This document outlines the design and implementation plan for a WhatsApp-style chat timeline for ATC radio transcriptions. This feature will enhance the existing transcription feature by displaying the content with more structured information including timestamps and speaker identification.

## Key Features

### 1. Timeline Interface
- A dedicated tab or panel in the frontend for viewing transcriptions in a timeline format
- Messages displayed as chat bubbles with different colors for different speakers
- Automatic scroll with the ability to jump to specific timestamps

### 2. Speaker Identification
- Different colors and labels for different speakers (controller, pilot, etc.)
- Automatic classification based on context and speech patterns when possible
- Manual override capability for corrections

### 3. Timestamp Integration
- Each message will have a precise timestamp showing when it occurred in the recording
- Timestamps will be clickable to jump to that part of the audio
- Timeline markers to show density of communication

### 4. Advanced Search and Filter
- Search within transcriptions for specific phrases or call signs
- Filter by speaker type
- Filter by time range

### 5. Export and Share
- Export the timeline transcription as a formatted text file
- Share specific sections with timestamps

## Implementation Plan

### Phase 1: Basic Timeline Display
- Add a new timeline container to the UI
- Parse existing transcriptions to extract potential speaker changes
- Display transcriptions with basic timestamp information

### Phase 2: Speaker Identification
- Implement speech pattern analysis to identify different speakers
- Add color coding and visual separation for different speakers
- Create a legend to identify speaker types

### Phase 3: Interactive Features
- Connect timeline to audio player for synchronized playback
- Implement clickable timestamps to navigate the audio
- Add search and filter capabilities

## UI Mockup

```
┌─────────────────────────────────────────────────────────────┐
│ ATC Radio Monitor & Transcriber                    🔍 Search │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Audio Player [▶️ Play] [⏹️ Stop] [⏪ Rewind] [⏩ Forward] │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌───────────────┬───────────────┬────────────────────────┐ │
│  │ Waveform      │ Transcription │ Timeline               │ │
│  ├───────────────┼───────────────┼────────────────────────┤ │
│  │               │               │                        │ │
│  │    /\/\/\/\   │ Full text     │ 00:12 Tower: Cessna    │ │
│  │   /        \  │ transcription │ 123, runway 27 clear   │ │
│  │  /          \ │ will be       │ for takeoff            │ │
│  │ /            \│ displayed     │                        │ │
│  │               │ here.         │ 00:18 Pilot: Cleared   │ │
│  │               │               │ for takeoff runway 27, │ │
│  │               │               │ Cessna 123             │ │
│  │               │               │                        │ │
│  │               │               │ 00:35 Tower: Delta     │ │
│  │               │               │ 456, descend and       │ │
│  │               │               │ maintain 3000          │ │
│  │               │               │                        │ │
│  └───────────────┴───────────────┴────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Technical Approach

1. **Transcript Parsing**:
   - Split existing transcriptions into sentence segments
   - Use NLP to analyze speech patterns and identify potential speaker changes
   - Extract timestamp information from the audio segments

2. **Speaker Classification**:
   - Build a simple classifier to identify:
     - Tower/Controller communications
     - Pilot responses
     - Other ground communications
   - Use context clues (runway mentions, clearance language, etc.)

3. **Timeline UI Components**:
   - Create a scrollable container for timeline messages
   - Design message bubbles with speaker indicators
   - Implement timestamp markers that link to audio positions

4. **Synchronization with Audio**:
   - Link timeline display to audio player
   - Allow bidirectional navigation (click timeline to move audio, and update timeline when audio plays)

## Integration with Existing System
This feature will build upon the existing transcription functionality, adding a more structured visualization layer without modifying the core transcription engine.