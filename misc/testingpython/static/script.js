document.addEventListener('DOMContentLoaded', function() {
    const vectorFieldImage = document.getElementById('vector-field');
    const particleAnimationImage = document.getElementById('particle-animation');
    const updateButton = document.getElementById('update-button');

    updateButton.addEventListener('click', function() {
        const uField = document.getElementById('u-field').value;
        const vField = document.getElementById('v-field').value;

        // Update the vector field image
        fetch('/vector-field', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ U: uField, V: vField })
        })
        .then(response => response.blob())
        .then(blob => {
            const url = URL.createObjectURL(blob);
            vectorFieldImage.src = url;
        })
        .catch(error => console.error('Error updating vector field:', error));

        // Update the particle animation image
        fetch('/particle-animation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ U: uField, V: vField })
        })
        .then(response => response.blob())
        .then(blob => {
            const url = URL.createObjectURL(blob);
            particleAnimationImage.src = url;
        })
        .catch(error => console.error('Error updating particle animation:', error));
    });
});


