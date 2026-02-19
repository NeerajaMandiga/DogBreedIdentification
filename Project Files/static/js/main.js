document.addEventListener('DOMContentLoaded', function(){
  const fileInput = document.getElementById('fileInput');
  const preview = document.getElementById('preview');
  const form = document.getElementById('uploadForm');
  const spinner = document.getElementById('spinner');
  const predictBtn = document.getElementById('predictBtn');

  function validFileType(file){
    const allowed = ['image/jpeg','image/png','image/gif','image/jpg'];
    return allowed.includes(file.type);
  }

  if(fileInput){
    fileInput.addEventListener('change', function(e){
      const file = e.target.files[0];
      if(!file){ preview.textContent='No image selected'; return; }
      if(!validFileType(file)){ alert('Invalid file type'); fileInput.value=''; preview.textContent='No image selected'; return; }
      const reader = new FileReader();
      reader.onload = function(ev){
        preview.innerHTML = '<img src="'+ev.target.result+'" style="max-width:100%;max-height:100%;border-radius:8px">';
      };
      reader.readAsDataURL(file);
    });
  }

  if(form){
    form.addEventListener('submit', function(){
      if(spinner) spinner.classList.remove('hidden');
      if(predictBtn) predictBtn.disabled = true;
    });
  }
});
