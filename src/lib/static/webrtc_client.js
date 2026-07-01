const sessionIdDict = {};

async function startStream(videoElementId, webrtcEndpoint) {
    console.log("Connecting...");

    const peerConnection = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
    });

    // Bind incoming track to the video element
    peerConnection.ontrack = (event) => {
        console.log("Track received from server!");
        const videoElement = document.getElementById(videoElementId);
        videoElement.srcObject = event.streams[0];
    };

    // Request video only (direction: recvonly) -> safe for insecure HTTP remote clients
    peerConnection.addTransceiver('video', { direction: 'recvonly' });

    // Generate client offer
    const offer = await peerConnection.createOffer();
    await peerConnection.setLocalDescription(offer);

    // CLIENT-SIDE ICE GATHERING: Wait for browser to discover paths completely
    await new Promise((resolve) => {
        if (peerConnection.iceGatheringState === 'complete') {
            resolve();
        } else {
            function checkState() {
                if (peerConnection.iceGatheringState === 'complete') {
                    peerConnection.removeEventListener('icegatheringstatechange', checkState);
                    resolve();
                }
            }
            peerConnection.addEventListener('icegatheringstatechange', checkState);
        }
    });

    console.log("Sending complete Offer to server...");

    // Send full Offer to NiceGUI / FastAPI
    const response = await fetch(webrtcEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            sdp: peerConnection.localDescription.sdp,
            type: peerConnection.localDescription.type,
            action: "connect"
        })
    });

    const answer = await response.json();
    console.log("Received Answer from server. Setting remote description...");
    sessionIdDict[videoElementId] = answer.sessionId;
    await peerConnection.setRemoteDescription(new RTCSessionDescription(answer));
    
    console.log("Connected");
}

async function stopStream(videoElementId, webrtcEndpoint) {
    
    // notify the server over HTTP to kill the tracks immediately
    currentSessionId = sessionIdDict[videoElementId]
    if (currentSessionId) {
        try {
            await fetch(webrtcEndpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    action: "close",
                    sessionId: currentSessionId
                })
            });
        } catch (e) {
            console.error("Failed to notify server of closure", e);
        }
        delete sessionIdDict[videoElementId]
    }

    // Clear the video element so it stops displaying the last frame
    const videoElement = document.getElementById(videoElementId);
    if (videoElement.srcObject) {
        videoElement.srcObject.getTracks().forEach(track => track.stop());
        videoElement.srcObject = null;
    }
    
    console.log("Stream stopped and connection closed locally.");
}