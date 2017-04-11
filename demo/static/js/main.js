/* global $ */
var row = 32
var col = 32
class Main {
    constructor() {
        this.canvas = document.getElementById('main');
        this.input = document.getElementById('input');
        this.recon = document.getElementById('reconstruct');
        this.canvas.width  = 16 * col + 1; // 16 * 28 + 1
        this.canvas.height = 16 * row + 1; // 16 * 28 + 1
        this.ctx = this.canvas.getContext('2d');
        this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
        this.canvas.addEventListener('mouseup',   this.onMouseUp.bind(this));
        this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
        this.initialize();
    }
    initialize() {
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.fillRect(0, 0, 16*col+1, 16*row+1);
        this.ctx.lineWidth = 1;
        this.ctx.strokeRect(0, 0, 16*col+1, 16*row+1);
        this.ctx.lineWidth = 0.05;
        for (var i = 0; i < row-1; i++) {
            this.ctx.beginPath();
            this.ctx.moveTo((i + 1) * 16,   0);
            this.ctx.lineTo((i + 1) * 16, 16*row+1);
            this.ctx.closePath();
            this.ctx.stroke();

            this.ctx.beginPath();
            this.ctx.moveTo(  0, (i + 1) * 16);
            this.ctx.lineTo(16*row+1, (i + 1) * 16);
            this.ctx.closePath();
            this.ctx.stroke();
        }
        this.drawInput();
    }
    onMouseDown(e) {
        this.canvas.style.cursor = 'default';
        this.drawing = true;
        this.prev = this.getPosition(e.clientX, e.clientY);
    }
    onMouseUp() {
        this.drawing = false;
        this.drawInput();
    }
    onMouseMove(e) {
        if (this.drawing) {
            var curr = this.getPosition(e.clientX, e.clientY);
            this.ctx.lineWidth = 16;
            this.ctx.lineCap = 'round';
            this.ctx.beginPath();
            this.ctx.moveTo(this.prev.x, this.prev.y);
            this.ctx.lineTo(curr.x, curr.y);
            this.ctx.stroke();
            this.ctx.closePath();
            this.prev = curr;
        }
    }
    getPosition(clientX, clientY) {
        var rect = this.canvas.getBoundingClientRect();
        return {
            x: clientX - rect.left,
            y: clientY - rect.top
        };
    }
    drawInput() {
        var ctx = this.input.getContext('2d');
        var img = new Image();
        img.onload = () => {
            var inputs = [];
            var small = document.createElement('canvas').getContext('2d');
            small.drawImage(img, 0, 0, img.width, img.height, 0, 0, col, row);
            var data = small.getImageData(0, 0, col, row).data;
            for (var i = 0; i < row; i++) {
                for (var j = 0; j < col; j++) {
                    var n = 4 * (i * row + j);
                    inputs[i * col + j] = (data[n + 0] + data[n + 1] + data[n + 2]) / 3;
                    ctx.fillStyle = 'rgb(' + [data[n + 0], data[n + 1], data[n + 2]].join(',') + ')';
                    ctx.fillRect(j * 5, i * 5, 5, 5);
                }
            }
            if (Math.min(...inputs) === 255) {
                draw_digit(this.recon, data);
                document.getElementById('answer').innerHTML = '';
                for(var i = 0; i < 72; i++) {
                    draw_digit(document.getElementById('digit_'+i), data);
                }


                return;
            }
        $.ajax({
            url: '/api/recon',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(inputs),
            success: (data) => {
                draw_digit(this.recon, data['result'][0]);
                document.getElementById('answer').innerHTML = data['pred'];
                for(var i = 0; i < 72; i++) {
                    draw_digit(document.getElementById('digit_'+i), data['result'][i+1]);
                }
            }
        });
        };
        img.src = this.canvas.toDataURL();
    }
}

function draw_digit(canvas, data) {
    var ctx = canvas.getContext('2d')
    for (var i = 0; i < row; i++) {
        for (var j = 0; j < col; j++) {
            ctx.fillStyle = 'rgb(' + [data[i*row+j], data[i*row+j], data[i*row+j]].join(',') + ')';
            ctx.fillRect(j * 5, i * 5, 5, 5);
        }
    }
}

class Main_hiragana {
    constructor() {
        this.canvas = document.getElementById('main_hiragana');
        this.input = document.getElementById('input_hiragana');
        this.recon = document.getElementById('reconstruct_hiragana');
        this.canvas.width  = 16 * col + 1; // 16 * 28 + 1
        this.canvas.height = 16 * row + 1; // 16 * 28 + 1
        this.ctx = this.canvas.getContext('2d');
        this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
        this.canvas.addEventListener('mouseup',   this.onMouseUp.bind(this));
        this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
        this.initialize();
    }
    initialize() {
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.fillRect(0, 0, 16*col+1, 16*row+1);
        this.ctx.lineWidth = 1;
        this.ctx.strokeRect(0, 0, 16*col+1, 16*row+1);
        this.ctx.lineWidth = 0.05;
        for (var i = 0; i < row-1; i++) {
            this.ctx.beginPath();
            this.ctx.moveTo((i + 1) * 16,   0);
            this.ctx.lineTo((i + 1) * 16, 16*row+1);
            this.ctx.closePath();
            this.ctx.stroke();

            this.ctx.beginPath();
            this.ctx.moveTo(  0, (i + 1) * 16);
            this.ctx.lineTo(16*row+1, (i + 1) * 16);
            this.ctx.closePath();
            this.ctx.stroke();
        }
        this.drawInput();
    }
    onMouseDown(e) {
        this.canvas.style.cursor = 'default';
        this.drawing = true;
        this.prev = this.getPosition(e.clientX, e.clientY);
    }
    onMouseUp() {
        this.drawing = false;
        this.drawInput();
    }
    onMouseMove(e) {
        if (this.drawing) {
            var curr = this.getPosition(e.clientX, e.clientY);
            this.ctx.lineWidth = 16;
            this.ctx.lineCap = 'round';
            this.ctx.beginPath();
            this.ctx.moveTo(this.prev.x, this.prev.y);
            this.ctx.lineTo(curr.x, curr.y);
            this.ctx.stroke();
            this.ctx.closePath();
            this.prev = curr;
        }
    }
    getPosition(clientX, clientY) {
        var rect = this.canvas.getBoundingClientRect();
        return {
            x: clientX - rect.left,
            y: clientY - rect.top
        };
    }
    drawInput() {
        var ctx = this.input.getContext('2d');
        var img = new Image();
        img.onload = () => {
            var inputs = [];
            var small = document.createElement('canvas').getContext('2d');
            small.drawImage(img, 0, 0, img.width, img.height, 0, 0, col, row);
            var data = small.getImageData(0, 0, col, row).data;
            for (var i = 0; i < row; i++) {
                for (var j = 0; j < col; j++) {
                    var n = 4 * (i * row + j);
                    inputs[i * col + j] = (data[n + 0] + data[n + 1] + data[n + 2]) / 3;
                    ctx.fillStyle = 'rgb(' + [data[n + 0], data[n + 1], data[n + 2]].join(',') + ')';
                    ctx.fillRect(j * 5, i * 5, 5, 5);
                }
            }
            if (Math.min(...inputs) === 255) {
                draw_digit(this.recon, data);
                document.getElementById('answer_hiragana').innerHTML = '';
                for(var i = 0; i < 72; i++) {
                    draw_digit(document.getElementById('hiragana_'+i), data);
                }


                return;
            }
        $.ajax({
            url: '/api/recon',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(inputs),
            success: (data) => {
                draw_digit(this.recon, data['result'][0]);
                document.getElementById('answer_hiragana').innerHTML = data['pred'];
                for(var i = 0; i < 72; i++) {
                    draw_digit(document.getElementById('hiragana_'+i), data['result'][i+1]);
                }
            }
        });
        };
        img.src = this.canvas.toDataURL();
    }
}

$(() => {
    var main = new Main();
    var main_hira = new Main_hiragana();
    $('#clear').click(() => {
        main.initialize();
    });
    $('#clear_hiragana').click(() => {
        main_hira.initialize();
    });
});
