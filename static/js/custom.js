document.querySelectorAll(".use-termynal").forEach(node => {
    node.style.display = "block";
    new Termynal(node, {
        lineDelay: 400
    });
});
const progressLiteralStart = "---> 100%";
const promptLiteralStart = "$ ";
const customPromptLiteralStart = "# ";
const termynalActivateClass = "termy";
let termynals = [];

function createTermynals() {
    document
        .querySelectorAll(`.${termynalActivateClass} .highlight`)
        .forEach(node => {
            const text = node.textContent;
            const lines = text.split("\n");
            const useLines = [];
            let buffer = [];
            function saveBuffer() {
                if (buffer.length) {
                    let isBlankSpace = true;
                    buffer.forEach(line => {
                        if (line) {
                            isBlankSpace = false;
                        }
                    });
                    dataValue = {};
                    if (isBlankSpace) {
                        dataValue["delay"] = 0;
                    }
                    if (buffer[buffer.length - 1] === "") {
                        // A last single <br> won't have effect
                        // so put an additional one
                        buffer.push("");
                    }
                    const bufferValue = buffer.join("<br>");
                    dataValue["value"] = bufferValue;
                    useLines.push(dataValue);
                    buffer = [];
                }
            }
            for (let line of lines) {
                if (line === progressLiteralStart) {
                    saveBuffer();
                    useLines.push({
                        type: "progress"
                    });
                } else if (line.startsWith(promptLiteralStart)) {
                    saveBuffer();
                    const value = line.replace(promptLiteralStart, "").trimEnd();
                    useLines.push({
                        type: "input",
                        value: value
                    });
                } else if (line.startsWith("// ")) {
                    saveBuffer();
                    const value = "# " + line.replace("// ", "").trimEnd();
                    useLines.push({
                        value: value,
                        class: "termynal-comment",
                        delay: 0
                    });
                } else if (line.startsWith(customPromptLiteralStart)) {
                    saveBuffer();
                    const promptStart = line.indexOf(promptLiteralStart);
                    if (promptStart === -1) {
                        console.error("Custom prompt found but no end delimiter", line)
                    }
                    const prompt = line.slice(0, promptStart).replace(customPromptLiteralStart, "")
                    let value = line.slice(promptStart + promptLiteralStart.length);
                    useLines.push({
                        type: "input",
                        value: value,
                        prompt: prompt
                    });
                } else {
                    buffer.push(line);
                }
            }
            saveBuffer();
            const div = document.createElement("div");
            node.replaceWith(div);
            const termynal = new Termynal(div, {
                lineData: useLines,
                noInit: true,
                lineDelay: 500
            });
            termynals.push(termynal);
        });
}

function loadVisibleTermynals() {
    termynals = termynals.filter(termynal => {
        if (termynal.container.getBoundingClientRect().top - innerHeight <= 0) {
            termynal.init();
            return false;
        }
        return true;
    });
}
window.addEventListener("scroll", loadVisibleTermynals);
createTermynals();
loadVisibleTermynals();
function setFavicon(isDark) {
    const favicon = document.querySelector('link[rel="icon"]');
    if (favicon) {
        const href = favicon.getAttribute('href');
        favicon.setAttribute('href', href.replace(isDark ? 'light' : 'dark', isDark ? 'dark' : 'light'));
    }
}

function setLogo(isDark) {
    const logo = document.querySelector('img[alt="logo"]');
    logo.classList.remove(isDark ? 'logo-light' : 'logo-dark');
    logo.classList.add(isDark ? 'logo-dark' : 'logo-light');
}
function ready(then) {
    if (['interactive', 'complete'].includes(document.readyState)) {
        then();
    } else {
        document.addEventListener('DOMContentLoaded', then);
    }
}
function checkDarkMode(fromEvent) {
    if (!window.localStorage.getItem('theme.lock')) {
        window.localStorage.setItem("theme.lock", "1");
        const fromBody = document.body.getAttribute('data-md-color-scheme');
        if (fromBody) {
            const isDark = fromBody === 'slate';
            if (fromEvent) {
                setFavicon(!isDark);
                setLogo(isDark);
            } else {
                setFavicon(isDark);
                setLogo(!isDark);
            }
        }
        setTimeout(() => {
            window.localStorage.removeItem("theme.lock");
        }, 200);
    }
}
ready(function() {
    if (window.localStorage.getItem('theme.lock')) {
        window.localStorage.removeItem("theme.lock");
    }
    checkDarkMode(false);
    document.querySelector('.md-header__option').addEventListener('click', function() {
        checkDarkMode(true);
    });
});
