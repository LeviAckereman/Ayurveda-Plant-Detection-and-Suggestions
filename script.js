// Animation for the homepage heading
document.addEventListener('DOMContentLoaded', () => {
    const heading = document.querySelector('.animate-fade-in');
    if (heading) {
      heading.style.opacity = '0';
      heading.style.transform = 'translateY(-20px)';
      setTimeout(() => {
        heading.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        heading.style.opacity = '1';
        heading.style.transform = 'translateY(0)';
      }, 100);
    }
  
    // Image upload handler
    const fileInput = document.getElementById('image-upload');
    if (fileInput) {
      fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
          console.log('Uploaded file:', file.name);
          // Add logic to preview/process the image
        }
      });
    }
  });