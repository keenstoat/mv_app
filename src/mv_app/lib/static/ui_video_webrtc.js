const peerConnections = {}

async function startStream(videoElementId, webrtcEndpoint) {
    console.log("Connecting...");

    const peerConnection = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
    });
    peerConnections[videoElementId] = peerConnection;

    // Bind incoming track to the video element
    peerConnection.ontrack = (event) => {
        console.log("Track received from server!");
        
        const videoElement = document.getElementById(videoElementId);
        videoElement.srcObject = event.streams[0];

        event.track.onended = () => {
            console.log("Remote video track ended.");
            stopStream(videoElementId, webrtcEndpoint);
        };
    };
    // add a handler to stop the stream when the peer closes it
    peerConnection.onconnectionstatechange = () => {
        const state = peerConnection.connectionState;
        console.log("WebRTC connection state:", state);
    
        if (state === "failed" || state === "closed" || state === "disconnected") {
            stopStream(videoElementId, webrtcEndpoint);
        }
    };

    peerConnection.on

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
        })
    });

    const answer = await response.json();
    console.log("Received Answer from server. Setting remote description...");
    await peerConnection.setRemoteDescription(new RTCSessionDescription(answer));

}

function stopStream(videoElementId, webrtcEndpoint) {

    console.log("Stream stopped requested...");
    
    // Clear the video element so it stops displaying the last frame
    const videoElement = document.getElementById(videoElementId);

    if (videoElement?.srcObject) {
        videoElement.srcObject.getTracks().forEach(track => track.stop());
        videoElement.srcObject = null;
    }

    // close the peer connection so the server stops streaming
    const peerConnection = peerConnections[videoElementId];

    if (peerConnection) {
        // Remove handlers before closing to avoid triggering cleanup again.
        peerConnection.onconnectionstatechange = null;
        peerConnection.ontrack = null;

        if (peerConnection.connectionState !== "closed") {
            peerConnection.close();
        }

        delete peerConnections[videoElementId];
    }

    console.log("Stream stopped and connection closed locally.");
}