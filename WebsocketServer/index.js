const WebSocket = require('ws');
 
const wss = new WebSocket.Server({
  port: 8080
});

wss.on('connection', function connection(ws) {
    ws.on('message', function incoming(data) {
        try {
            JSON.parse(data);
        } catch( err) {
            return;
        }
            console.log('received message: ' + data);
            wss.clients.forEach(client => client.send(data));
    });
   
});
