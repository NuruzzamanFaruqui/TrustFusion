#!/bin/bash
echo "[+] Copying systemd service file..."
sudo cp edge-device-trustfusion.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable edge-device-trustfusion.service
sudo systemctl start edge-device-trustfusion.service
echo "[✓] Edge device service started and enabled on boot."
