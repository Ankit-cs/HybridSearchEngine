document.getElementById("query").addEventListener("keyup", function(event) {
    if (event.key === "Enter") {
        search();
    }
});

async function search() {

    let query = document.getElementById("query").value;
    let k = document.getElementById("k").value;

    let status = document.getElementById("status");
    let resultsDiv = document.getElementById("results");

    if (!query.trim()) {
        alert("Please enter a query");
        return;
    }

    status.innerText = "Searching...";
    resultsDiv.innerHTML = "";

    let url = `http://localhost:8000/api/v1/search?q=${query}&k=${k}`;

    try {
        let response = await fetch(url);
        let data = await response.json();

        showResults(data);

    } catch (error) {
        status.innerText = "Error connecting to API";
        console.log(error);
    }
}

function showResults(data) {

    document.getElementById("status").innerText =
        `Found ${data.results.length} results in ${data.took_ms} ms`;

    let resultsDiv = document.getElementById("results");
    resultsDiv.innerHTML = "";

    data.results.forEach(r => {

        let div = document.createElement("div");
        div.className = "result-card";

        // ---- ADD IT HERE ----
        let link = r.url;
        // ---------------------

        div.innerHTML = `
            <div class="result-title">
                <a href="${link}" target="_blank">${r.title}</a>
            </div>

            <div class="snippet">
                ${r.snippet}
            </div>

            <div class="score">
                <span class="doc-id">ID: ${r.doc_id}</span> | Score: ${r.score}
            </div>
        `;


        resultsDiv.appendChild(div);
    });
}
