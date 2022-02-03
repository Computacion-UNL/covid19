const dropArea = document.querySelector(".drop-area");
const target = document.querySelector(".target");
const dropText = dropArea.querySelector("h2");
const buttonDA = dropArea.querySelector("button");
const input = dropArea.querySelector("#input-file");
const imagePreview = target.querySelector("#img-preview");
const diagnos19 = document.querySelector("#diagnos19");
const newDiagnos19 = document.querySelector('#newDiagnos19');
const upload_BAR = document.querySelector("#progressBar");
// Servicio de Cloudinary
const cloudinary_URL = 'https://api.cloudinary.com/v1_1/jr98/image/upload';
const cloudinary_UPLOAD_PRESET = 'hmhymhxk';
const covidDg = document.querySelector('#covidDg');
const normalDg = document.querySelector('#normalDg');
const chartPB = document.querySelector("#progressPB");
const conoraImg = document.querySelector('#coronaImg');
let files;
let acc_score;
    buttonDA.addEventListener("click", (e) => {
        input.click();
    });

input.addEventListener("change", async (e) => {
    files = input.files;
    dropArea.classList.add("active");
    const trFile = e.target.files[0];
    const formData = new FormData();
    formData.append('file', trFile);
    formData.append('upload_preset', cloudinary_UPLOAD_PRESET);
    const resp = await axios.post(cloudinary_URL, formData, {
        Headers: {
            'Content-Type': 'multipart/form-data'
        },
        onUploadProgress(e) {
            console.log(Math.round(e.loaded * 100));
            progress_status = (e.loaded * 100) / e.total;
            upload_BAR.removeAttribute("hidden");
            upload_BAR.setAttribute('value', progress_status);
        }
    });
    console.log(resp);
    path_img = resp.data.secure_url;
    imagePreview.src = resp.data.secure_url;
    modeloCall(path_img);
    target.removeAttribute("hidden");
    newDiagnos19.removeAttribute("hidden");
    dropArea.classList.remove('active');
});

function showFiles(files) {
    if (files.length === undefined) {
        processFile(files);
    } else {
        for (const file of files) {
            processFile(file);
        }
    }
}

function processFile(file) {
    const docType = file.type;
    const validExtension = ['image/png'];

    if (validExtension.includes(docType)) {
        //Archivo valido
        const fileReader = new FileReader();


        fileReader.addEventListener("load", (e) => {
            const fileUrl = fileReader.result;
            const id = `file-${Math.random().toString(32).substring(7)}`;
            const image = `
                <div id = "${id}" class = "file-container" style =" border: none">
                    <div class = "status">
                        <span>${file.name}</span>
                    </div>
                </div>`;
            const html = document.querySelector('#preview').innerHTML;
            document.querySelector('#preview').innerHTML = image + html;
        });
        fileReader.readAsDataURL(file);
    } else {
        //Archico invalifo
        alert('No es un archivo válido');

    }
}

function modeloCall(path_img) {
    diagnos19.addEventListener("click", async (e) => {
        url = path_img;
        const diagnostic = await axios.post('http://127.0.0.1:3000/image/addres_img?url=' + url)
            .then(function (response) {
                console.log(response);
                showDiagnostic(response.data.diagnostico);
                acc_score = response.data.accuracy_score;
                acc_score = (acc_score * 100).toFixed(3);
                conoraImg.style.display = 'none';
                //Chart 



                const DATA_COUNT = 2;
                const NUMBER_CFG = {
                    count: DATA_COUNT,
                    min: 0,
                    max: 100
                };

                new Chart(chartPB, {
                    type: 'doughnut',
                    data: {
                        labels: ['Precisión ' + '(' + Math.round(acc_score) + '%)', 'Confusión ' + '(' + Math.round((100 - acc_score)) + '%)'],
                        datasets: [{
                            data: [acc_score, (100 - acc_score)],
                            backgroundColor: [
                                'rgba(54, 162, 235, 1)',
                                'rgba(255, 99, 132, 1)',
                            ],
                            borderColor: [
                                'rgba(54, 162, 235, 0.2)',
                                'rgba(255, 99, 132, 0.2)',
                            ]
                        }]
                    },
                    showDatapoints: true,
                    options: {
                        responsive: true,
                        plugins: {
                            legend: {
                                position: 'top',
                            },
                            labels: {
                                render: 'label',
                                fontStyle: 'normal',
                                fontSize: 20,
                                fontColor: '#111111',
                                fontFamily: 'Arial',
                                arc: true,
                            },
                            datalabels: {
                                backgroundColor: function (context) {
                                    return context.dataset.backgroundColor;
                                },
                                color: '#111111',
                                font: {
                                    weight: 'bold',
                                    size: 18
                                },
                                padding: 4,
                            }
                        },
                        title: {
                            display: true,
                            text: 'Precisión del diagnóstico',
                            fontSize: 20
                        },
                        animation: {
                            animateScale: true,
                            animateRotate: true
                        },
                    },
                });
                //Fin Chart
            })
            .catch(function (error) {
                console.log(error);
            })
            .then(function () {});
    });
}

function showDiagnostic(diagnosticRst) {
    if (diagnosticRst == 0) {
        covidDg.removeAttribute("hidden");
    } else if (diagnosticRst == 1) {
        normalDg.removeAttribute("hidden");
    } else {
        alert('La radiografía ingreseda no se puede clasificar');
    }

}

newDiagnos19.addEventListener("click", (e) => {
    location.reload();
    newDiagnos19.style.display = 'none';
}, input.click());