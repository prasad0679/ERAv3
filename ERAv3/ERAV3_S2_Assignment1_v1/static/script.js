document.querySelectorAll('input[name="animal"]').forEach(radio => {
    radio.addEventListener('change', function() {
        const animalImage = document.getElementById('animal-image');
        animalImage.innerHTML = `<img src="/static/images/${this.value}.jpg" alt="${this.value}">`;
    });
});

function uploadFile() {
    const fileInput = document.getElementById('file-upload');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a file to upload.');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById('file-info').innerHTML = `<p>Error: ${data.error}</p>`;
        } else {
            document.getElementById('file-info').innerHTML = `
                <p>File Name: ${data.name}</p>
                <p>File Size: ${data.size}</p>
                <p>File Type: ${data.type}</p>
            `;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('file-info').innerHTML = '<p>An error occurred while uploading the file.</p>';
    });
}

function changeBackground() {
    const select = document.getElementById('background-select');
    const selectedBackground = select.value;
    document.body.style.backgroundImage = `url('/static/images/backgrounds/${selectedBackground}.jpg')`;
}
