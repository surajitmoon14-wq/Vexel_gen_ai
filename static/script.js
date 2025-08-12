document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('image-form');
    const promptInput = document.getElementById('prompt-input');
    const generateBtn = document.getElementById('generate-btn');
    const loader = document.getElementById('loader');
    const imageContainer = document.getElementById('image-container');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const prompt = promptInput.value;
        if (!prompt) {
            alert('Please enter a prompt.');
            return;
        }

        // Show loader and disable button
        loader.style.display = 'block';
        generateBtn.disabled = true;
        imageContainer.innerHTML = ''; // Clear previous image

        try {
            const response = await fetch('/generate-image', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ prompt: prompt }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Something went wrong');
            }

            const data = await response.json();

            // Create an image element from the base64 string
            const img = document.createElement('img');
            img.src = `data:image/png;base64,${data.image_b64}`;
            imageContainer.appendChild(img);

        } catch (error) {
            console.error('Error:', error);
            imageContainer.innerHTML = `<p style="color: #ff4d4d;">Error: ${error.message}</p>`;
        } finally {
            // Hide loader and enable button
            loader.style.display = 'none';
            generateBtn.disabled = false;
        }
    });
});